# SN101 Competitive Miner — Code Architecture

This document maps the current miner (`master` / `b0-floor` baseline) from the
validator request down to the scoring model it optimizes against.

Entry chain: `tasks/competitive_sn101.py:solve_problem` →
`CompetitiveMiner(...)` → `generate_tags(post)` →
`tasks/sn101_reference/core/competitive_miner.py`.

The `b1`–`b6` experiment branches only swap the **selection logic** in
diagram 2 (the middle block). Diagrams 1, 3, and 4 are identical across all
branches.

---

## 1. End-to-end flow: validator → miner → score

```mermaid
flowchart TD
    V["Validator sends TaskEnvelope<br/>(payload.text = post)"] --> SP["solve_problem(envelope, runtime)<br/>tasks/competitive_sn101.py"]
    SP --> CM["CompetitiveMiner(n_tags=3, timeout_sec)"]
    CM --> GT["generate_tags(post)"]
    GT --> TAGS["{ tags: [t1, t2, t3] }"]
    TAGS --> RET["return to validator"]

    RET -.submitted with all miners.-> SC["score_answers()<br/>competitive_sn101_scoring.py"]
    SC --> TS["TagScorer.score(post, all_responses)"]
    TS --> FORMULA["per tag:<br/>0.60·consensus + 0.40·(validity×diversity)<br/>miner_score = aggregate(top-3)"]

    classDef mine fill:#1e3a5f,stroke:#4a90d9,color:#fff
    classDef score fill:#3f2a1e,stroke:#d98a4a,color:#fff
    class CM,GT,TAGS mine
    class SC,TS,FORMULA score
```

---

## 2. Inside `generate_tags` (baseline pipeline)

```mermaid
flowchart TD
    A["post.strip()"] -->|empty| Z0["return []"]
    A --> B["_clean_post_text(post)<br/>strip URLs / @mentions / punctuation"]
    B --> C["_candidate_pool(clean_post, raw_post)"]

    C --> D["_select_span_only_tags()<br/>keep only 'strong' entity-like spans"]
    D --> E["_finalize_tags() → span_tags"]
    E --> F{"len(span_tags) ≥ 3 ?"}
    F -->|yes| G["return span_tags[:3]<br/>FAST PATH · no LLM"]

    F -->|no| H{"_has_api_key() ?"}
    H -->|yes| I["_generate_with_llm(post, span_candidates)<br/>OpenAI chat/completions → _parse_tags"]
    H -->|no| J["llm_tags = []"]
    I --> K
    J --> K["_merge_candidates(span, llm)"]

    K --> L["_select_final_tags(pool)"]
    L --> L1["_rerank_for_consensus()"]
    L1 --> L2["_pick_by_embedding_centroid()<br/>MiniLM centrality + diversity gate"]
    L2 --> L3["_fill_from_spans() if short"]
    L3 --> M{"empty?"}
    M -->|yes| N["_fallback_tags(llm, spans)"]
    M -->|no| O
    N --> O["_ensure_n_tags(_finalize_tags(selected))<br/>guarantee 3 grounded tags (floor fix)"]
    O --> P["return tags[:3]"]

    classDef fast fill:#1e4a2e,stroke:#4ad97a,color:#fff
    classDef llm fill:#4a1e3f,stroke:#d94aa0,color:#fff
    class G fast
    class I,K llm
```

---

## 3. `_candidate_pool` internals (candidate generation + filtering)

```mermaid
flowchart LR
    subgraph SRC["sources (clean_post + raw_post)"]
        S1["build_spans()<br/>noun-ish spans"]
        S2["_expand_span_chunks()<br/>prefix/tail chunks"]
        S3["Capitalized words<br/>[A-Z][...]"]
        S4["#hashtags"]
        S5["@mentions"]
        S6["_token_ngram_candidates()<br/>1–2 grams (fallback)"]
    end

    SRC --> ADD["add(raw)"]
    ADD --> NORM["smart_tag() → normalize<br/>tag_wordmap brand map"]
    NORM --> TRIM["_trim_tag_edges()<br/>strip edge stop-words"]
    TRIM --> F1["_is_valid_format()"]
    F1 --> F2["_is_coherent_tag()<br/>reject fragments / contractions / dupes"]
    F2 --> F3["_is_junk_tag()<br/>reject slugs (o98n0hfax8), consonant runs"]
    F3 --> F4["_is_low_value_tag()<br/>reject generic / fluff"]
    F4 --> POOL["_sort_candidate_tags()<br/>prefer ~2-word tags"]

    classDef filt fill:#3a2e1e,stroke:#d9b84a,color:#fff
    class F1,F2,F3,F4 filt
```

---

## 4. The scoring model the miner optimizes against

```mermaid
flowchart TD
    POST["post"] --> CTX["build_scoring_context<br/>(all miners' tags + spans)"]

    CTX --> CON["ConsensusScorer<br/>embed → AgglomerativeClustering<br/>support × centroid-proximity"]
    CTX --> VAL["ValidityScorer<br/>max(sim_post, sim_span, lexical)<br/>× format → tier {0,.3,.6,1}"]
    CTX --> DIV["DiversityScorer<br/>inter-tag dissimilarity"]

    CON --> T["tag_score =<br/>0.60·consensus + 0.40·(validity×diversity)"]
    VAL --> T
    DIV --> T
    T --> AGG["aggregate_miner_score(top-3)"]

    CON -. "60% · depends on OTHER miners (unknowable a priori)" .-> NOTE1[" "]
    VAL -. "deterministic from post" .-> NOTE2[" "]
    DIV -. "deterministic from your 3 tags" .-> NOTE2

    classDef hidden fill:#4a1e1e,stroke:#d94a4a,color:#fff
    classDef known fill:#1e4a2e,stroke:#4ad97a,color:#fff
    class CON hidden
    class VAL,DIV known
```

---

## Strategy branches (selection-logic variants of diagram 2)

| Branch | Strategy | LLM | Embeddings | Core idea |
|--------|----------|-----|------------|-----------|
| `b0-floor` | Baseline balanced | gated | yes | span fast-path → LLM gate → centroid select |
| `b1` | Simplest extractive | no | no | first-mentioned post spans |
| `b2` | Extractive centroid | no | yes | tags nearest embedding centroid of candidates |
| `b3` | LLM abstractive | always | no | trust model's human-like tags, span backfill |
| `b4` | Topic salience | no | no | rank by position + entity signal + repetition |
| `b5` | Validity-max | no | yes | rank by validator grounding `max(sim_post, sim_span, lexical)` |
| `b6` | Ensemble (complex) | always | yes | Borda blend of validity + consensus + salience + LLM |

All variants share the same `_candidate_pool` (diagram 3), the same
`_ensure_n_tags` floor, and are scored by the same model (diagram 4).
