"""Bittensor miner entry point."""

import asyncio
import signal
import time
from typing import Any, Tuple

from ._bt import require_bittensor
from .chain.runtime import ChainRuntime
from .chain.settings import build_config
from .observability import configure_observability, log_validator_task, task_scope
from .protocol import TaskEnvelope
from .tasks import TaskRegistry, TaskServerClient, build_task_registry


class SolverMiner:
    def __init__(self, *, config: Any | None = None, registry: TaskRegistry | None = None):
        self.bt = require_bittensor()
        self.config = config or build_config("miner")
        self.runtime = ChainRuntime(self.config, role="miner")
        self.registry = registry or self._task_registry()
        self.client = self._task_server_client()
        self.axon = self.runtime.axon()
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self.should_exit = False
        self._configure_observability()

    def _configure_observability(self) -> None:
        enable_task_log = bool(getattr(self.config.miner, "enable_task_log", True))
        enable_cost_log = bool(getattr(self.config.miner, "enable_cost_log", True))
        log_dir = str(getattr(self.config.miner, "log_dir", "") or "").strip()
        if not log_dir:
            log_dir = str(self.config.neuron.storage_dir)
        configure_observability(
            log_dir,
            enable_task_log=enable_task_log,
            enable_cost_log=enable_cost_log,
        )
        self.bt.logging.info(
            f"miner observability log_dir={log_dir} "
            f"task_log={enable_task_log} cost_log={enable_cost_log}"
        )

    async def forward(self, synapse: TaskEnvelope) -> TaskEnvelope:
        started = time.perf_counter()
        dendrite = getattr(synapse, "dendrite", None)
        validator_hotkey = getattr(dendrite, "hotkey", None)
        validator_uid = self._uid_for_hotkey(validator_hotkey)
        tweet_id = str(dict(synapse.payload or {}).get("tweet_id", "") or "")
        post_text = str(dict(synapse.payload or {}).get("text", "") or "")
        success = True
        error_text = ""
        tags: list[str] = []
        try:
            with task_scope(
                task_id=str(synapse.task_id or ""),
                task_kind=str(synapse.task_kind or ""),
                validator_hotkey=validator_hotkey,
                validator_uid=validator_uid,
                netuid=int(self.config.netuid),
                miner_uid=int(self.runtime.uid),
                tweet_id=tweet_id,
            ):
                handler = self.registry.handler_for(synapse.task_kind)
                synapse.answer = handler.solve_problem(synapse, self.runtime)
            raw_tags = synapse.answer.get("tags") if isinstance(synapse.answer, dict) else None
            if isinstance(raw_tags, list):
                tags = [str(tag) for tag in raw_tags if isinstance(tag, str)]
        except Exception as exc:
            success = False
            error_text = f"{type(exc).__name__}: {exc}"
            synapse.answer = {}
        elapsed = time.perf_counter() - started
        self.bt.logging.info(
            f"MINER_SOLVED_TASK task={synapse.task_id} kind={synapse.task_kind} "
            f"validator_uid={validator_uid} validator={str(validator_hotkey or '')[:16]} "
            f"elapsed={elapsed:.3f}s tags={tags} answer_keys={list(synapse.answer)}"
        )
        log_validator_task(
            task_id=str(synapse.task_id or ""),
            task_kind=str(synapse.task_kind or ""),
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            netuid=int(self.config.netuid),
            miner_uid=int(self.runtime.uid),
            post_text=post_text,
            tweet_id=tweet_id,
            tags=tags,
            elapsed_sec=elapsed,
            success=success,
            error=error_text or None,
        )
        return synapse

    def _uid_for_hotkey(self, hotkey: str | None) -> int | None:
        if not hotkey:
            return None
        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if hotkey not in hotkeys:
            return None
        return hotkeys.index(hotkey)

    async def blacklist(self, synapse: TaskEnvelope) -> Tuple[bool, str]:
        dendrite = getattr(synapse, "dendrite", None)
        hotkey = getattr(dendrite, "hotkey", None)
        if not hotkey:
            allow_empty = bool(getattr(self.config.miner, "allow_empty_hotkey", False))
            return (not allow_empty), "missing caller hotkey"

        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if hotkey not in hotkeys:
            if bool(getattr(self.config.blacklist, "allow_non_registered", False)):
                return False, "unregistered caller allowed by config"
            return True, "caller is not registered"

        uid = hotkeys.index(hotkey)
        if bool(getattr(self.config.blacklist, "force_validator_permit", False)):
            permits = getattr(self.runtime.metagraph, "validator_permit", [])
            if uid >= len(permits) or not bool(permits[uid]):
                return True, "caller has no validator permit"
        return False, "accepted"

    async def priority(self, synapse: TaskEnvelope) -> float:
        dendrite = getattr(synapse, "dendrite", None)
        hotkey = getattr(dendrite, "hotkey", None)
        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if not hotkey or hotkey not in hotkeys:
            return 0.0
        stakes = getattr(self.runtime.metagraph, "S", [])
        uid = hotkeys.index(hotkey)
        return float(stakes[uid]) if uid < len(stakes) else 0.0

    def run(self) -> None:
        self.runtime.ensure_registered()
        hide_public_axon = bool(getattr(self.config.miner, "hide_axon_from_metagraph", False))
        if hide_public_axon:
            hidden_port = int(getattr(self.config.axon, "external_port", None) or self.axon.external_port or 8092)
            self.runtime.serve_axon(self.axon, chain_ip="0.0.0.0", chain_port=hidden_port)
            self.bt.logging.info(
                f"miner published hidden metagraph axon 0.0.0.0:{hidden_port}; "
                "real endpoint will be announced to the task server"
            )
        else:
            self.runtime.serve_axon(self.axon)
        self.axon.start()
        self.bt.logging.info(f"miner serving at block {self.runtime.block}")
        self._announce_private_axon_at_startup()

        def _request_shutdown(signum: int, _frame: Any) -> None:
            self.bt.logging.info(f"miner shutdown requested (signal {signum})")
            self.should_exit = True

        signal.signal(signal.SIGTERM, _request_shutdown)
        signal.signal(signal.SIGINT, _request_shutdown)

        try:
            while not self.should_exit:
                self.runtime.sync_metagraph()
                time.sleep(12)
        except KeyboardInterrupt:
            self.bt.logging.info("miner interrupted")
        finally:
            self.axon.stop()

    def _task_server_client(self) -> TaskServerClient | None:
        url = str(getattr(self.config.task_server, "url", "") or "").strip()
        if not url or bool(getattr(self.config.miner, "axon_to_public_metagraph", False)):
            return None
        return TaskServerClient(
            url,
            timeout=float(getattr(self.config.task_server, "timeout", 30.0)),
            verify_ssl=bool(getattr(self.config.task_server, "verify_ssl", True)),
        )

    def _task_registry(self) -> TaskRegistry:
        miner_module = str(getattr(getattr(self.config, "task", None), "miner_module", "") or "")
        override_modules = [
            module.strip()
            for module in miner_module.split(",")
            if module.strip()
        ]
        return build_task_registry(override_modules=override_modules)

    def _announce_private_axon_at_startup(self) -> None:
        if self.client is None:
            return

        try:
            result = asyncio.run(self._announce_private_axon_once())
        except Exception as exc:
            self.bt.logging.warning(
                "private miner axon startup announcement failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        if not result.get("accepted") or not result.get("updated", True):
            self.bt.logging.warning(
                "private miner axon announcement returned unexpected response "
                f"response={result}"
            )
        else:
            self.bt.logging.info(
                "MINER_PRIVATE_AXON_ANNOUNCED "
                f"netuid={int(self.config.netuid)} block={self.runtime.block}"
            )

    async def _announce_private_axon_once(self) -> dict[str, Any]:
        if self.client is None:
            return {"accepted": False, "updated": False}
        info = self.axon.info()
        ip = getattr(info, "ip", None)
        port = getattr(info, "port", None)
        if not ip or not port:
            raise ValueError("private axon announcement requires external ip and port")
        return await self.client.announce_miner_axon(
            wallet=self.runtime.wallet,
            netuid=int(self.config.netuid),
            uid=int(self.runtime.uid),
            block=self.runtime.block,
            ip=str(ip),
            port=int(port),
            ip_type=None,
            protocol=int(getattr(info, "protocol", 4) or 4),
            version=int(getattr(info, "version", 0) or 0),
        )

    def __enter__(self) -> "SolverMiner":
        self._task = asyncio.get_event_loop().run_in_executor(None, self.run)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.should_exit = True


def main() -> None:
    SolverMiner().run()


if __name__ == "__main__":
    main()
