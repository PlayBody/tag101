"""Competitive Tag101 miner tuned for the subnet scoring formula.

Optimizes for:
  TagScore = 0.6 * consensus + 0.4 * validity * diversity

Strategy:
  1. Extract span/entity candidates (same logic the validator validity scorer uses).
  2. Use span-only answers when the post has enough clear entities (consensus-first).
  3. Call the LLM only when spans are thin (short/RT posts or sparse entities).
  4. Pick final tags with embedding centroid ranking (MiniLM, same model as validators).
  5. Enforce diversity threshold aligned with the validator diversity scorer (0.55).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import typing
from pathlib import Path
from urllib.request import Request, urlopen

_PREPROCESSING_PATH = Path(__file__).resolve().parent / "scoring" / "preprocessing.py"
_PREPROCESSING_SPEC = importlib.util.spec_from_file_location(
    "tag101_scoring_preprocessing",
    _PREPROCESSING_PATH,
)
assert _PREPROCESSING_SPEC and _PREPROCESSING_SPEC.loader
_PREPROCESSING = importlib.util.module_from_spec(_PREPROCESSING_SPEC)
sys.modules[_PREPROCESSING_SPEC.name] = _PREPROCESSING
_PREPROCESSING_SPEC.loader.exec_module(_PREPROCESSING)
build_spans = _PREPROCESSING.build_spans
normalize_tag = _PREPROCESSING.normalize_tag

from .tag_wordmap import smart_tag
from .tag_wordmap import _EXPLICIT_MAP as _BRAND_MAP

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_N_TAGS = 3
DEFAULT_TIMEOUT_SEC = 60
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIVERSITY_SIMILARITY_THRESHOLD = 0.55
SHORT_POST_CHAR_LIMIT = 48
_LAZY_EMBEDDER: typing.Any | None = None
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "for",
        "with",
        "on",
        "in",
        "to",
        "of",
        "and",
        "or",
        "at",
        "by",
        "from",
        "as",
        "it",
        "its",
        "this",
        "that",
        "today",
        "new",
        "better",
        "used",
        "lol",
        "now",
        "just",
        "yes",
        "yeah",
        "here",
        "there",
        "very",
        "really",
        "also",
        "still",
        "even",
        "about",
        "into",
        "over",
        "under",
        "up",
        "out",
        "off",
        "how",
        "what",
        "when",
        "where",
        "who",
        "why",
        "which",
        "want",
        "wants",
        "using",
        "uses",
        "use",
        "like",
        "have",
        "has",
        "had",
        "get",
        "gets",
        "make",
        "makes",
        "tell",
        "tells",
        "talking",
        "talk",
        "create",
        "creates",
        "explore",
        "explores",
    }
)
_EDGE_STOP_WORDS = _STOP_WORDS | frozenset(
    {
        "your",
        "our",
        "their",
        "its",
        "his",
        "her",
        "my",
        "we",
        "you",
        "they",
        "them",
        "those",
        "these",
        "some",
        "any",
        "all",
        "one",
        "two",
        "three",
        "first",
        "last",
        "next",
        "then",
        "than",
        "so",
        "if",
        "but",
        "not",
        "no",
        "yes",
        "via",
        "per",
        "vs",
        "s",
        "re",
        "ve",
        "ll",
        "d",
        "m",
        "t",
        "here",
        "there",
        "free",
        "for",
        "building",
        "build",
        "built",
        "hope",
        "see",
        "full",
        "most",
        "more",
        "less",
        "best",
        "ever",
    }
)
_FRAGMENT_VERBS = frozenset(
    {
        "explore",
        "explores",
        "building",
        "build",
        "uses",
        "use",
        "using",
        "want",
        "wants",
        "hope",
        "hopes",
        "see",
        "tell",
        "talk",
        "talking",
        "create",
        "creates",
        "like",
        "get",
        "gets",
        "make",
        "makes",
        "had",
        "has",
        "have",
        "was",
        "were",
        "been",
        "got",
        "did",
        "ever",
    }
)
_CONTRACTION_TOKENS = frozenset({"ve", "re", "ll", "d", "m", "t", "s", "nt", "ive"})

_URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(com|org|net|io|ai|co)\b)",
    re.IGNORECASE,
)
_URL_IN_TEXT = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"@\w+")
_TCO_FRAGMENT = re.compile(r"^co [a-z0-9]{4,}$")
_JUNK_TOKENS = frozenset(
    {
        "http",
        "https",
        "www",
        "t",
        "co",
        "amp",
        "rt",
        "pic",
        "twitter",
        "x",
    }
)

# Umbrella tags that validate but cluster poorly against specific posts.
_GENERIC_TAGS = frozenset(
    {
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "technology",
        "innovation",
        "startup",
        "business",
        "news",
        "tech",
        "personal development",
        "productivity",
        "productivity tips",
        "ai news",
        "ai industry",
        "ai industry concerns",
        "ai research",
        "ai model",
        "ai models",
        "ai tools",
        "ai products",
        "weekly roundup",
        "top stories",
        "business advice",
        "side project",
        "incorporation",
        "building projects",
        "current trends",
        "regulatory changes",
        "forward-looking",
        "furthermore",
        "historically",
        "belief systems",
        "social conformity",
    }
)

# Vague/sentiment tags that validate but rarely form large consensus clusters.
_FLUFF_TAGS = frozenset(
    {
        "excited",
        "anniversary",
        "belated reply",
        "cool paper",
        "highly suggest",
        "because we",
        "everyone why",
        "can follow",
        "all who",
        "email me",
        "eas",
    }
)

TAGGING_SYSTEM_PROMPT = """\
You tag social posts for a decentralized labeling network.
Return ONLY a JSON array of lowercase strings.

Rules (strict):
- Output exactly {n_tags} tags.
- Each tag: 1 to 5 words, lowercase, no URLs, no hashtags, no @mentions.
- Tags must describe entities, topics, products, events, or themes in the post.
- Tags must be relevant; do not invent facts not supported by the post.
- Tags must be semantically different from each other (not paraphrases).
- Prefer concise noun phrases miners would agree on (e.g. "openai", "gpt-5", "ai regulation").
- Prefer 1-2 word tags when they capture the topic; avoid umbrella terms like "artificial intelligence" when a specific entity or product is named.
- Match common short names from the post (brands, people, products, tickers, events).
- Tags are compared across many miners — pick wording others would likely use too, not creative paraphrases.
"""

TAGGING_USER_PROMPT = """\
Post:
{post}

Helpful extracted phrases from the post (use when relevant, do not copy blindly):
{span_hints}

Return a JSON array of exactly {n_tags} tags.
"""


class CompetitiveMiner:
    """Scoring-aware miner. Drop-in replacement for ReferenceMiner."""

    # Identifies the tag-generation strategy this build ships. Overridden on
    # each experiment branch (b1..b6) so deployed UIDs are self-identifying.
    STRATEGY = "b0-baseline-balanced"

    def __init__(
        self,
        api_key: str = OPENAI_KEY,
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_MODEL,
        n_tags: int = DEFAULT_N_TAGS,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.n_tags = n_tags
        self.timeout_sec = timeout_sec

    def generate_tags(self, post: str) -> list[str]:
        post = post.strip()
        if not post:
            return []

        clean_post = self._clean_post_text(post)
        span_candidates = self._candidate_pool(clean_post, raw_post=post)

        span_selected = self._select_span_only_tags(
            clean_post, span_candidates, raw_post=post
        )
        span_tags = self._finalize_tags(span_selected)
        if len(span_tags) >= self.n_tags:
            return span_tags[: self.n_tags]

        llm_tags: list[str] = []
        if self._has_api_key():
            llm_post = clean_post if len(clean_post) >= 20 else post
            try:
                llm_tags = self._generate_with_llm(llm_post, span_candidates)
            except RuntimeError:
                llm_tags = []

        pool = self._merge_candidates(span_candidates, llm_tags)
        selected = self._select_final_tags(pool, span_candidates)
        if not selected:
            selected = self._fallback_tags(llm_tags, span_candidates, self.n_tags)
        return self._ensure_n_tags(
            self._finalize_tags(selected),
            clean_post,
            span_candidates,
        )[: self.n_tags]

    def _ensure_n_tags(
        self,
        tags: list[str],
        clean_post: str,
        span_candidates: list[str],
    ) -> list[str]:
        out = list(tags)
        if len(out) >= self.n_tags:
            return out
        ranked = self._rerank_for_consensus(
            [tag for tag in span_candidates if not self._is_low_value_tag(tag)],
            span_candidates,
        )
        out = self._fill_from_spans(out, ranked, self.n_tags)
        if len(out) >= self.n_tags:
            return out
        for tag in self._token_ngram_candidates(clean_post):
            if len(out) >= self.n_tags:
                break
            if tag in out or not self._is_valid_format(tag):
                continue
            if not self._is_coherent_tag(tag):
                continue
            if out and not self._is_diverse_enough(tag, out):
                continue
            out.append(tag)
        if len(out) >= self.n_tags:
            return out
        # Last resort: never ship fewer than n_tags. Missing tags score 0 on BOTH
        # validity and diversity, so a plausible single word beats an empty slot.
        # Diversity gating is dropped here (only deduped) since this only fires on
        # short/sparse posts where we already exhausted the ranked pools.
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", clean_post.lower()):
            if len(out) >= self.n_tags:
                break
            if len(token) < 3 or token in _STOP_WORDS or token in _JUNK_TOKENS:
                continue
            cand = self._trim_tag_edges(smart_tag(token))
            if not cand or cand in out:
                continue
            if self._is_junk_tag(cand) or self._is_low_value_tag(cand):
                continue
            out.append(cand)
        return out

    def _finalize_tags(self, tags: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in tags:
            tag = smart_tag(raw)
            tag = self._trim_tag_edges(tag)
            if not tag or tag in seen:
                continue
            if not self._is_valid_format(tag):
                continue
            if not self._is_coherent_tag(tag):
                continue
            out.append(tag)
            seen.add(tag)
        return out

    def _select_span_only_tags(
        self,
        clean_post: str,
        span_candidates: list[str],
        *,
        raw_post: str = "",
    ) -> list[str]:
        """Skip the LLM when strong entity-like spans cover the post."""
        strong = [
            tag
            for tag in span_candidates
            if self._is_strong_candidate(tag, clean_post, raw_post=raw_post)
        ]
        if len(strong) < self.n_tags:
            return []
        ranked = self._rerank_for_consensus(strong, span_candidates)
        selected = self._pick_by_embedding_centroid(ranked, self.n_tags)
        if len(selected) < self.n_tags:
            selected = self._select_diverse_tags(ranked, self.n_tags)
        if len(selected) < self.n_tags:
            return []
        if any(self._is_low_value_tag(tag) for tag in selected):
            return []
        return selected

    def _should_call_llm(self, clean_post: str, span_candidates: list[str]) -> bool:
        coherent = [tag for tag in span_candidates if self._is_coherent_tag(tag)]
        if len(clean_post) <= SHORT_POST_CHAR_LIMIT:
            return len(coherent) < self.n_tags
        strong = sum(
            1 for tag in coherent if self._is_strong_candidate(tag, clean_post)
        )
        if strong >= self.n_tags + 1:
            return False
        return True

    def _is_strong_candidate(self, tag: str, post: str, *, raw_post: str = "") -> bool:
        if self._is_low_value_tag(tag) or not self._is_valid_format(tag):
            return False
        if not self._is_coherent_tag(tag):
            return False
        if len(tag.split()) >= 2:
            return True
        source = f"{post}\n{raw_post}"
        if re.search(rf"@\s*{re.escape(tag)}\b", source, re.IGNORECASE):
            return True
        if re.search(rf"#\s*{re.escape(tag)}\b", source, re.IGNORECASE):
            return True
        compact = re.sub(r"[\s_\-]+", "", tag.lower())
        if compact in _BRAND_MAP:
            return True
        if re.search(rf"#\s*{re.escape(compact)}\b", source, re.IGNORECASE):
            return True
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9+\-]{1,}\b", source):
            if normalize_tag(match.group(0)) == tag:
                return True
        if len(tag) >= 5 and re.search(rf"\b{re.escape(tag)}\b", source, re.IGNORECASE):
            return True
        return False

    def _select_final_tags(
        self,
        pool: list[str],
        span_candidates: list[str],
    ) -> list[str]:
        if not pool:
            return []
        ranked = self._rerank_for_consensus(pool, span_candidates)
        selected = self._pick_by_embedding_centroid(ranked, self.n_tags)
        if len(selected) < self.n_tags:
            selected = self._fill_from_spans(selected, ranked, self.n_tags)
        return selected[: self.n_tags]

    @classmethod
    def _get_embedder(cls) -> typing.Any | None:
        global _LAZY_EMBEDDER
        if _LAZY_EMBEDDER is not None:
            return _LAZY_EMBEDDER
        try:
            from .scoring.tag_scorer import get_embedding_model

            _LAZY_EMBEDDER = get_embedding_model(EMBEDDING_MODEL)
        except Exception:
            return None
        return _LAZY_EMBEDDER

    def _pick_by_embedding_centroid(self, candidates: list[str], n: int) -> list[str]:
        if not candidates:
            return []
        if len(candidates) <= n:
            return candidates[:n]

        model = self._get_embedder()
        if model is None:
            return self._select_diverse_tags(candidates, n)

        import numpy as np

        vectors = model.encode(
            candidates,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        centrality = vectors @ vectors.T
        np.fill_diagonal(centrality, 0.0)
        counts = max(len(candidates) - 1, 1)
        mean_sim = centrality.sum(axis=1) / float(counts)

        order = sorted(
            range(len(candidates)),
            key=lambda idx: (-float(mean_sim[idx]), candidates[idx]),
        )

        selected: list[str] = []
        selected_idx: list[int] = []
        for idx in order:
            if len(selected) >= n:
                break
            tag = candidates[idx]
            if self._is_low_value_tag(tag):
                continue
            if selected and not self._is_embedding_diverse(idx, selected_idx, vectors):
                continue
            selected.append(tag)
            selected_idx.append(idx)
        return selected

    def _is_embedding_diverse(
        self,
        candidate_idx: int,
        selected_indices: list[int],
        vectors: typing.Any,
    ) -> bool:
        for idx in selected_indices:
            if float(vectors[candidate_idx] @ vectors[idx]) >= DIVERSITY_SIMILARITY_THRESHOLD:
                return False
        return True

    def _is_low_value_tag(self, tag: str) -> bool:
        if tag in _GENERIC_TAGS or tag in _FLUFF_TAGS:
            return True
        tokens = tag.split()
        if len(tokens) == 1 and tokens[0] in _FLUFF_TAGS:
            return True
        if len(tokens) == 1 and tokens[0] in _STOP_WORDS:
            return True
        return False

    def _fallback_tags(
        self,
        llm_tags: list[str],
        span_candidates: list[str],
        n: int,
    ) -> list[str]:
        """Ensure we never return empty tags when the post had content."""
        out: list[str] = []
        seen: set[str] = set()
        for tag in [*llm_tags, *span_candidates]:
            if tag in seen or self._is_junk_tag(tag):
                continue
            if not self._is_valid_format(tag):
                continue
            if not self._is_coherent_tag(tag):
                continue
            out.append(tag)
            seen.add(tag)
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _clean_post_text(post: str) -> str:
        text = _URL_IN_TEXT.sub(" ", post)
        text = _MENTION_PATTERN.sub(" ", text)
        text = re.sub(r"[\n\r]+", ". ", text)
        text = re.sub(r"[^\w\s\-+#]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or post.strip()

    @staticmethod
    def _trim_tag_edges(tag: str) -> str:
        words = tag.split()
        while words and words[0] in _EDGE_STOP_WORDS:
            words.pop(0)
        while words and words[-1] in _EDGE_STOP_WORDS:
            words.pop()
        return " ".join(words)

    def _is_coherent_tag(self, tag: str) -> bool:
        """Reject sentence fragments, mention tails, and HTML/punctuation junk."""
        if not tag:
            return False
        if "@" in tag or "," in tag or "=" in tag:
            return False
        if re.search(r"[()\[\]{}]", tag):
            return False
        if re.search(r"(^|\s)['\"]", tag) or tag.endswith((".", ":", "-", "'")):
            return False
        trimmed = self._trim_tag_edges(tag)
        if not trimmed or trimmed != tag:
            return False
        tokens = trimmed.split()
        if not tokens:
            return False
        if tokens[0] in _EDGE_STOP_WORDS or tokens[-1] in _EDGE_STOP_WORDS:
            return False
        if tokens[0] in _FRAGMENT_VERBS:
            return False
        if any(token in _CONTRACTION_TOKENS for token in tokens):
            return False
        if len(tokens) >= 3 and any(token in _EDGE_STOP_WORDS for token in tokens[1:-1]):
            return False
        if len(tokens) >= 2 and sum(1 for token in tokens if token in _FRAGMENT_VERBS) >= 2:
            return False
        if len(tokens) >= 3 and tokens[0] in _FRAGMENT_VERBS:
            return False
        if len(tokens) == 1 and len(tokens[0]) <= 2 and tokens[0] not in {"ai", "ml", "llm", "etf", "xrp", "btc", "xai"}:
            return False
        # Broken possessive / contraction fragments from cleaned punctuation.
        if tokens[-1] in {"s", "re", "ve", "ll", "d", "m", "t"}:
            return False
        if len(tokens) >= 2 and tokens[-2] in {"world", "meta", "your", "our"} and tokens[-1] == "s":
            return False
        if any(tokens[i] == tokens[i + 1] for i in range(len(tokens) - 1)):
            return False
        if any(re.fullmatch(r"h[1-6]", token) for token in tokens):
            return False
        if len(tokens) >= 2 and any(
            len(token) == 1 and token not in {"x"} for token in tokens
        ):
            return False
        return True

    @staticmethod
    def _expand_span_chunks(span: str) -> list[str]:
        words = span.split()
        if len(words) <= 5:
            return [span]
        # Avoid sliding windows — they create junk like "kids about" / "uses gray".
        chunks = [" ".join(words[: min(3, len(words))])]
        if len(words) > 3:
            tail = " ".join(words[-3:])
            if tail != chunks[0]:
                chunks.append(tail)
        if words[0] not in _STOP_WORDS:
            chunks.append(words[0])
        return list(dict.fromkeys(chunks))

    def _token_ngram_candidates(self, post: str) -> list[str]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", post.lower())
            if token not in _STOP_WORDS and token not in _JUNK_TOKENS
        ]
        candidates: list[str] = []
        seen: set[str] = set()
        for size in (1, 2):
            for index in range(0, max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[index : index + size])
                tag = smart_tag(phrase)
                if not tag or tag in seen or not self._is_valid_format(tag):
                    continue
                seen.add(tag)
                candidates.append(tag)
        return self._sort_candidate_tags(candidates)

    def _candidate_pool(self, post: str, *, raw_post: str = "") -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def add(raw: str) -> None:
            tag = smart_tag(raw)
            tag = self._trim_tag_edges(tag)
            if not tag or tag in seen or not self._is_valid_format(tag):
                return
            if not self._is_coherent_tag(tag):
                return
            words = tag.split()
            if words[0] in _STOP_WORDS or words[-1] in _STOP_WORDS:
                return
            seen.add(tag)
            candidates.append(tag)

        span_sources = [post]
        if raw_post and raw_post != post:
            span_sources.append(raw_post)

        for source in span_sources:
            for span in build_spans(source):
                for chunk in self._expand_span_chunks(span):
                    words = chunk.split()
                    if 1 <= len(words) <= 5:
                        add(chunk)

            for match in re.finditer(r"\b[A-Z][A-Za-z0-9+\-]{1,}\b", source):
                add(match.group(0))

            for match in re.finditer(r"#([A-Za-z][A-Za-z0-9_]{1,})", source):
                add(match.group(1))

            for match in re.finditer(r"@([A-Za-z][A-Za-z0-9_]{1,})", source):
                add(match.group(1))

        if len(candidates) < self.n_tags:
            for tag in self._token_ngram_candidates(post):
                if tag not in seen:
                    seen.add(tag)
                    candidates.append(tag)

        return self._sort_candidate_tags(candidates)

    def _has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key not in {"your openai key", "your_openai_key"})

    def _generate_with_llm(self, post: str, span_candidates: list[str]) -> list[str]:
        hints = ", ".join(span_candidates[:12]) if span_candidates else "(none)"
        system_prompt = TAGGING_SYSTEM_PROMPT.format(n_tags=self.n_tags)
        user_prompt = TAGGING_USER_PROMPT.format(
            post=post,
            span_hints=hints,
            n_tags=self.n_tags,
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        content = self._call_chat_completions(payload)
        return self._parse_tags(content)

    def _call_chat_completions(self, payload: dict[str, typing.Any]) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        request = Request(
            url=endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_text = response.read().decode("utf-8")
            data = json.loads(response_text)
            self._record_usage(data, payload.get("model", self.model))
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    def _record_usage(self, data: dict[str, typing.Any], model: str) -> None:
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return
        try:
            from tag101.observability import record_api_cost

            event = record_api_cost(
                model=str(model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            if event is not None:
                print(
                    "MINER_API_COST "
                    f"model={event['model']} "
                    f"tokens={event['total_tokens']} "
                    f"cost_usd={event['cost_usd']:.8f} "
                    f"day_total_usd={event['cost_usd_day_total']:.6f} "
                    f"task_id={event.get('task_id') or ''}",
                    flush=True,
                )
        except Exception:
            return

    def _parse_tags(self, raw_content: str) -> list[str]:
        text = raw_content.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            ).strip()

        try:
            tags = json.loads(text)
        except json.JSONDecodeError:
            tags = re.split(r"[,\n|]+", text)

        if not isinstance(tags, list):
            return []

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in tags:
            if not isinstance(value, str):
                continue
            tag = smart_tag(value)
            if not tag or tag in seen or not self._is_valid_format(tag):
                continue
            seen.add(tag)
            cleaned.append(tag)
        return cleaned

    def _merge_candidates(self, span_candidates: list[str], llm_tags: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for tag in [*self._sort_candidate_tags(span_candidates), *llm_tags]:
            if tag in seen or self._is_low_value_tag(tag):
                continue
            seen.add(tag)
            merged.append(tag)
        return merged

    def _rerank_for_consensus(
        self,
        candidates: list[str],
        span_candidates: list[str],
    ) -> list[str]:
        if not candidates:
            return []
        span_set = set(span_candidates)
        post_tokens = set()
        for span in span_candidates:
            post_tokens.update(self._tokenize(span))

        def rank_key(tag: str) -> tuple[int, int, int, int, int, str]:
            words = tag.split()
            generic = 1 if self._is_low_value_tag(tag) else 0
            span_hit = 0 if self._matches_spans(tag, span_set, post_tokens) else 1
            length_penalty = abs(len(words) - 2)
            long_penalty = 1 if len(words) >= 4 else 0
            single_word = 0 if len(words) == 1 else 1
            return (generic, span_hit, length_penalty, long_penalty, single_word, tag)

        return sorted(candidates, key=rank_key)

    def _matches_spans(
        self,
        tag: str,
        span_set: set[str],
        post_tokens: set[str],
    ) -> bool:
        if tag in span_set:
            return True
        tag_tokens = self._tokenize(tag)
        if tag_tokens & post_tokens:
            return True
        for span in span_set:
            if tag in span or span in tag:
                return True
        return False

    @staticmethod
    def _sort_candidate_tags(candidates: list[str]) -> list[str]:
        def rank(tag: str) -> tuple[int, int]:
            words = len(tag.split())
            return (abs(words - 2), words)

        return sorted(candidates, key=rank)

    def _select_diverse_tags(self, candidates: list[str], n: int) -> list[str]:
        if not candidates:
            return []
        if len(candidates) <= n:
            return [tag for tag in candidates if not self._is_low_value_tag(tag)][:n] or candidates[:n]

        selected: list[str] = []
        for tag in candidates:
            if len(selected) >= n:
                break
            if self._is_low_value_tag(tag) and selected:
                continue
            if not selected or self._is_diverse_enough(tag, selected):
                selected.append(tag)
        return selected

    def _fill_from_spans(
        self,
        selected: list[str],
        span_candidates: list[str],
        n: int,
    ) -> list[str]:
        out = list(selected)
        seen = set(out)
        for tag in span_candidates:
            if len(out) >= n:
                break
            if tag in seen:
                continue
            if out and not self._is_diverse_enough(tag, out):
                continue
            out.append(tag)
            seen.add(tag)
        return out

    def _is_diverse_enough(self, tag: str, selected: list[str]) -> bool:
        """Mirror diversity scorer thresholds using token Jaccard as a fast proxy."""
        for other in selected:
            if self._token_jaccard(tag, other) >= DIVERSITY_SIMILARITY_THRESHOLD:
                return False
        return True

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized: set[str] = set()
        for token in tokens:
            if token.endswith("ies") and len(token) > 4:
                normalized.add(f"{token[:-3]}y")
            elif token.endswith("s") and len(token) > 3:
                normalized.add(token[:-1])
            else:
                normalized.add(token)
        return normalized

    def _token_jaccard(self, left: str, right: str) -> float:
        a = self._tokenize(left)
        b = self._tokenize(right)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _is_valid_format(self, tag: str) -> bool:
        normalized = tag.strip()
        if not normalized:
            return False
        if self._is_junk_tag(normalized):
            return False
        if self._is_low_value_tag(normalized):
            return False
        if _URL_PATTERN.search(normalized):
            return False
        token_count = len(re.findall(r"[a-z0-9]+", normalized.lower()))
        if token_count < 1 or token_count > 5:
            return False
        if re.fullmatch(r"[\d\s.,:/%+\-]+", normalized):
            return False
        if re.fullmatch(r"[^\w\s]+", normalized, flags=re.UNICODE):
            return False
        if not self._is_coherent_tag(normalized):
            return False
        compact = [ch for ch in normalized if not ch.isspace()]
        if not compact:
            return False
        special_count = sum(
            1
            for ch in compact
            if not ch.isalnum() and ch not in {"-", "_", "&", "/", "+"}
        )
        return (special_count / len(compact)) <= 0.3

    @staticmethod
    def _is_junk_tag(tag: str) -> bool:
        lowered = tag.lower().strip()
        if not lowered:
            return True
        if "#" in lowered:
            return True
        if _TCO_FRAGMENT.match(lowered):
            return True
        if "http" in lowered or "www." in lowered:
            return True
        tokens = re.findall(r"[a-z0-9]+", lowered)
        if not tokens:
            return True
        if all(token in _STOP_WORDS for token in tokens):
            return True
        if len(tokens) == 1 and tokens[0] in _STOP_WORDS:
            return True
        if all(token in _JUNK_TOKENS for token in tokens):
            return True
        if sum(1 for token in tokens if token in _JUNK_TOKENS) >= max(1, len(tokens) // 2):
            return True
        if tokens[0] in _JUNK_TOKENS or tokens[-1] in _JUNK_TOKENS:
            return True
        # Reject tags that are mostly short noise tokens from truncated RT posts.
        alpha_tokens = [token for token in tokens if len(token) >= 3 or token in {"ai", "ml", "llm", "etf", "xrp", "btc"}]
        if not alpha_tokens:
            return True
        if len(tokens) >= 2 and all(len(token) <= 2 for token in tokens):
            return True
        for token in tokens:
            if token.isdigit():
                return True
            # t.co slugs and opaque handle fragments (e.g. qdosgfm5tw, gvrurodf0e)
            if len(token) >= 6 and re.fullmatch(r"[a-z0-9]+", token):
                has_alpha = any(ch.isalpha() for ch in token)
                has_digit = any(ch.isdigit() for ch in token)
                if sum(ch in "aeiou" for ch in token) == 0:
                    return True
                # letter+digit slug where a digit is followed by a letter, e.g.
                # o98n0hfax8 / od6x3iytti / kgnkbb3nmu (real brands like gpt4,
                # claude3 keep digits trailing and are not flagged).
                if has_alpha and has_digit and re.search(r"\d[a-z]", token):
                    return True
            # opaque consonant-only handle fragments (e.g. pmqqrhhiek); 6+ run is
            # far beyond any real English word (max ~5).
            if len(token) >= 7 and re.search(r"[bcdfghjklmnpqrstvwxz]{6,}", token):
                return True
        return False
