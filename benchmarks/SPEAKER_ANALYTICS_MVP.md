# Speaker Analytics MVP

Quick harness to spot-check whether speaker-aware prompts are an improvement over plain transcript text.

## What it does
- Loads a tiny sample set from `benchmarks/data/speaker_analytics_samples.jsonl`.
- Builds turn metadata + `speaker_stats` (no audio features required).
- Renders two prompts:
  - Baseline: segment text only.
  - Enriched: `to_speaker_summary()` + `to_turn_view()` with timestamps/audio cues.
- Summarizes both (LLM if available, extractive fallback otherwise).
- Judges baseline vs enriched (LLM judge if available, lexical overlap fallback otherwise).

## Run it
```bash
python benchmarks/eval_speaker_utility.py
# or with a model (requires OPENAI_API_KEY and openai>=1.x)
python benchmarks/eval_speaker_utility.py --llm-model gpt-4o-mini --output-json /tmp/speaker_eval.json --output-md /tmp/speaker_eval.md
```

Useful flags:
- `--samples PATH` – JSONL with `transcript_path`, `reference_summary`, optional `speaker_labels`.
- `--limit N` – cap how many samples to run.
- `--judge-model` – override judge model (defaults to `--llm-model`).

## Sample set
- `support_call.json`: agent/customer billing issue.
- `design_review.json`: PM/engineer/designer discussing release scope.
- `pricing_negotiation.json`: seller/buyer price objection and contract terms.
- `escalation_handoff.json`: agent/lead/customer with an escalation and timeline demand.
- `project_retro.json`: manager/engineer/QA assigning retro action items.
- Reference summaries live alongside the transcripts in the JSONL file.

## v1 conclusion
- On 5 calls with `gpt-4o-mini` (lexical fallback offline), enriched prompts were preferred **5/5** times; net gains are modest and most visible on attribution-sensitive asks (objections, escalations, action owners).
- Analytics are most useful for **targeted prompts** (“who objected?”, “who dominated the call?”) and **QA/coaching** reviews—not generic summaries.

## Notes
- OpenAI calls are optional; without them the harness uses deterministic overlap scoring so you can run it offline.
- Add your own transcripts by appending to `benchmarks/data/speaker_analytics_samples.jsonl` (schema v2 JSONs work fine).

## Results (2025-12-01)
- Config: same instructions for baseline/enriched (enriched adds speaker stats + turn cues); lexical fallback (OPENAI_API_KEY not set), samples from `benchmarks/data/speaker_analytics_samples.jsonl`.
- Outcome: enriched 5/5 (100%), baseline 0/5 (0%), ties 0 — lexical judge preferred enriched summaries on the tiny offline set; expect different outcomes once real LLMs are used.

### Notes
- The tiny set is meant to catch regressions, not claim quality. Swap in your own calls and LLMs for a more realistic view.
- Judge instructions now emphasize speaker attribution, objections/resolutions, and action ownership over brevity.
