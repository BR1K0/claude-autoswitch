# Handoff vs compaction, across many chats

Does an ACPMH handoff beat Claude Code's compaction on chats other than the one it was tuned on, with any model
taking over? This test reruns the comparison on more and messier chats, three sizes and three incoming models.
Run in November 2026, on a Claude subscription through Claude Code (no API key).

## Setup

- **Chats:** 9 real Claude Code chats: 4 coding sessions with many file edits, 1 planning + coding, 1 research
  (data analysis homework), 1 app testing and debugging, 1 planning + benchmarking, and 1 long messy chat (topic
  changes, reversed decisions, long code dumps). The chats are private, so only totals are published.
- **Sizes:** each chat cut at 200k real tokens. Only 2 chats reach 500k and 1 reaches 1M; the "1M" point is cut at
  920k, the largest history that still fits a 1M window with the instructions.
- **Questions:** Opus 5.5 writes 40 questions per point from the full history, 10 in each category: decisions
  (including reversed ones), early instructions still in force, file and code state, and what's left to do. Same
  prompt for every chat.
- **Methods**, each answered by Haiku 5.5, Sonnet 5.5 and Opus 5.5:
  - **Re-read:** the whole history (the reference, 100%).
  - **Compaction:** Opus writes a summary with a Claude Code-style compaction prompt; the incoming model reads it.
  - **Handoff:** the ACPMH handoff exactly as Autoswitch ships it (word-for-word journal, no API-billed fact
    extraction; v4 under 300k, v5b above), plus the journal lookup tools part 2 has.
  - **Handoff + lookup rule:** the same, with the rule added in 0.3.8: search the journal before answering
    "unknown". Run for Haiku and Sonnet, the models that were skipping the journal.
- **Grading:** Opus grades each answer against the gold answer (1 / 0.5 / 0), blind to the method. Scores are
  shown as % of the same model re-reading the whole chat.
- **Usage:** API-price equivalent of the tokens used per switch, as reported by Claude Code. Compaction includes
  writing the summary (Opus reads the whole history once); the handoff costs nothing to build.

## Results (median across chats, range in brackets)

| Size | Taking over | Re-read | Compaction | Handoff | Handoff + lookup rule |
|---|---|---|---|---|---|
| 200k (9 chats) | Opus | 100%, $1.87 | 80% (62-91%), $2.03 | **97% (94-100%)**, $1.51 | - |
| | Sonnet | 100%, $0.92 | 82% (70-90%), $1.95 | 88% (55-96%), $0.46 | **92% (85-96%)**, $0.66 |
| | Haiku | 100%, $0.24 | 79% (71-94%), $1.88 | 89% (51-92%), $0.12 | **91% (0-92%)**, $0.16 |
| 500k (2 chats) | Opus | 100%, $4.27 | 73% (71-74%), $4.40 | **95% (92-97%)**, $1.52 | - |
| | Sonnet | 100%, $2.11 | 77% (75-79%), $4.32 | 68% (61-75%), $0.39 | **78% (73-83%)**, $0.50 |
| | Haiku | 100%, $0.54 | **75% (67-82%)**, $4.25 | 64% (64-65%), $0.04 | 72% (68-76%), $0.10 |
| ~1M (1 chat) | Opus | 100%, $7.77 | 75%, $7.90 | **100%**, $1.82 | - |
| | Sonnet | 100%, $3.87 | 70%, $7.85 | 68%, $0.33 | **76%**, $0.55 |
| | Haiku | 100%, $0.98 | 71%, $7.79 | 66%, $0.02 | **82%**, $0.09 |

Per-chat totals (anonymised, with per-category scores): [results/handoff_vs_compaction.json](results/handoff_vs_compaction.json).

## What it shows

- **Opus taking over (a rollover):** the handoff remembered more than compaction in every chat at every size
  (92-100% vs 62-91%) and used 26-77% less.
- **Sonnet and Haiku skipped the journal.** Opus looked things up 7-30 times per chat; Haiku and Sonnet almost
  never did, and answered "unknown" instead (Sonnet: 15 of 40 in the worst chat). The lookup rule fixed most of it:
  Sonnet's worst 200k chat went from 55% to 85%, and the worst chat overall from 19.5-21.5 to 35 of 40.
- **With the rule, Sonnet and Haiku beat compaction at 200k and roughly tie it at 500k and above,** for 66-99%
  less usage.
- **Haiku is still unreliable:** in one chat it answered "unknown" to all 40 questions without looking anything up
  (the 0%). That run is counted, not rerun.
- **Compaction is weakest on "what's left to do"**, and loses decisions and early instructions on long chats.
- **Re-reading is still the most accurate** wherever the history fits, and with Haiku it is cheap ($0.24 at 200k,
  $0.98 at ~1M).

## Limits

- 500k and 1M rest on 2 chats and 1 chat, both from the same author; 200k has 9.
- One person's chats, one question writer and one grader (both Opus); each answer graded once.
- Compaction here is a Claude Code-style prompt run by Opus, not Claude Code's own `/compact`.
- The lookup rule was given to the incoming model as an instruction in the test; in 0.3.8 it is part of the
  handoff text itself.

## Rerun it

`bench/handoff_vs_compaction.py` (needs Claude Code signed in and `claude-agent-sdk`; it refuses to run if a call
would bill an API key). List your own chats in a JSON file:

```bash
echo '[{"id": "my-chat", "kind": "coding", "transcript": "/path/to/chat.jsonl"}]' > chats.json
uv run --with claude-agent-sdk==0.2.182 python3 bench/handoff_vs_compaction.py \
    --chats chats.json --sizes 200000,500000,1000000 --out bench/out/hvc
uv run --with claude-agent-sdk==0.2.182 python3 bench/handoff_vs_compaction.py \
    --chats chats.json --models haiku,sonnet --arms handoff_lookup --out bench/out/hvc
python3 bench/handoff_vs_compaction.py --report --out bench/out/hvc
```

Every step is cached, so a stopped run continues where it left off. Expect about $15 of usage per chat at 200k,
$35 at 500k and $70 at 1M (API-price equivalent).
