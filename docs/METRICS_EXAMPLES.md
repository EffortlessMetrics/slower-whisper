# Metrics Examples

Quick-start KPIs you can compute from `speaker_stats` without building a UI.

## Run

```bash
python examples/metrics/call_kpis.py transcripts/meeting.json
```

## What it prints

- Talk ratio per speaker (percentage of total talk time).
- Longest monologue (duration + snippet).
- Question rate per speaker (questions / turns).
- Interruptions started/received.
- Sentiment signal per speaker (if sentiment is present in `speaker_stats`).

## Where to use these KPIs

- Talk ratio → sales call coaching
- Longest monologue → agent training/rambling detection
- Question rate → discovery quality
- Interruptions → friction/rapport issues

## Notes

- The script will derive turns and `speaker_stats` on the fly if missing.
- Treat sentiment as a coarse signal, not a diagnosis.
- Use this as a starting point for dashboards or QA scoring; adapt thresholds to your domain.
