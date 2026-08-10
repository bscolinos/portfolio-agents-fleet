---
name: singlestore-research-writer
description: The ONE sanctioned way to persist research results (hypotheses, experiments, findings, activity, Aura queries) to SingleStore. Use it for every result you produce so all rows are uniform across the fleet. Never write SQL directly or invent fields.
---

# SingleStore Research Writer

You are a research agent on a fleet. Every result you produce — hypotheses,
backtest experiments, findings, activity-log steps, hypothesis-status changes,
and Aura Analyst queries — MUST be persisted through this templated write tool.
It validates and normalizes every row (controlled `strategy_family` enum,
canonical metric keys, auto-generated id + timestamp + Qwen embedding), so the
fleet's data stays uniform no matter which agent writes it. **Never** connect to
the database directly, never emit SQL, never invent columns.

## How to call it

The tool is an HTTP service on the host, reachable from your sandbox at
`http://host.openshell.internal:11510`. Use `curl`. Every call returns a JSON
receipt `{"ok": true, "id": "...", "table": "..."}` (or `{"ok": false, "error": "..."}`
with HTTP 422 if you sent an invalid value — read the error and fix your payload).

Discover the exact schema any time:
```bash
curl -s http://host.openshell.internal:11510/tools | jq .
```

Write a record (replace `<tool>` and the JSON body):
```bash
curl -s -X POST http://host.openshell.internal:11510/tool/<tool> \
  -H 'Content-Type: application/json' -d '<json-payload>'
```

## Tools (and their required fields)

`strategy_family` is a controlled enum — use EXACTLY one of:
`equal_weight, momentum, mean_reversion, vol_target, low_vol, factor, risk_parity, regime`.

Canonical metric keys (put any you have into `metrics`): `sharpe, ann_vol,
ann_return, max_drawdown, turnover, cvar_95, win_rate, n_rebalances,
benchmark_sharpe, beats_benchmark`. Anything else is dropped.

- `write_activity` — required: `agent_id`, `phase` (one of
  START|RECALL|HYPOTHESIS|EXPERIMENT|FINDING|ANALYST|END|ERROR|NOTE). optional: `detail`, `task_id`.
- `write_hypothesis` — required: `agent_id`, `statement`, `strategy_family`.
  optional: `rationale`, `params`, `confidence` (0..1), `task_id`. Returns the `id` to pass along.
- `write_experiment` — required: `agent_id`, `strategy_family`, `params`.
  optional: `metrics`, `hypothesis_id`, `task_id`, `universe`, `method`
  (python-backtest|aura-analyst|llm-analysis), `engine`, `lookback_start`,
  `lookback_end`, `status` (ok|failed), `error`.
- `write_finding` — required: `agent_id`, `content`, `strategy_family`.
  optional: `title`, `kind` (finding|insight|caveat|next_step), `metrics`,
  `experiment_id`, `hypothesis_id`, `task_id`, `importance` (0..1), `tags`.
- `set_hypothesis_status` — required: `hypothesis_id`, `status`
  (open|testing|supported|rejected|inconclusive).
- `record_analyst_query` — required: `agent_id`, `question`. optional:
  `generated_sql`, `row_count`, `answer`, `latency_ms`, `status`, `task_id`.

## Required arc for every research task

1. `write_activity` phase=START (include your `agent_id` and the `task_id`).
2. Recall relevant prior findings; `write_activity` phase=RECALL.
3. `write_hypothesis` — a falsifiable, quantitative claim. Keep its `id`.
4. Backtest, then `write_experiment` with the metrics (always vs the 1/N
   benchmark, net of turnover cost). Keep its `id`.
5. (Optional) Ask Aura Analyst, then `record_analyst_query`.
6. `write_finding` — quantitative, honest (say plainly if it does NOT beat 1/N),
   link `experiment_id`/`hypothesis_id`, and include a next step.
7. `set_hypothesis_status` consistent with the result.
8. `write_activity` phase=END.

## Example

```bash
# hypothesis
HID=$(curl -s -X POST http://host.openshell.internal:11510/tool/write_hypothesis \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"research-01","statement":"12-1 momentum (top 20) beats 1/N net of cost in trending regimes","strategy_family":"momentum","confidence":0.6,"task_id":"task-abc"}' | jq -r .id)

# experiment
EID=$(curl -s -X POST http://host.openshell.internal:11510/tool/write_experiment \
  -H 'Content-Type: application/json' \
  -d "{\"agent_id\":\"research-01\",\"strategy_family\":\"momentum\",\"params\":{\"lookback_days\":252,\"skip_days\":21,\"top_n\":20},\"metrics\":{\"sharpe\":0.79,\"ann_vol\":0.19,\"max_drawdown\":-0.35,\"turnover\":0.9,\"benchmark_sharpe\":0.74,\"beats_benchmark\":true},\"hypothesis_id\":\"$HID\",\"task_id\":\"task-abc\"}" | jq -r .id)

# finding
curl -s -X POST http://host.openshell.internal:11510/tool/write_finding \
  -H 'Content-Type: application/json' \
  -d "{\"agent_id\":\"research-01\",\"strategy_family\":\"momentum\",\"content\":\"12-1 momentum returned Sharpe 0.79 vs 0.74 for 1/N, but 90% turnover erodes the edge net of cost; concentrated in trending regimes. Next: test a turnover cap.\",\"metrics\":{\"sharpe\":0.79,\"beats_benchmark\":true},\"experiment_id\":\"$EID\",\"hypothesis_id\":\"$HID\",\"task_id\":\"task-abc\"}"

curl -s -X POST http://host.openshell.internal:11510/tool/set_hypothesis_status \
  -H 'Content-Type: application/json' -d "{\"hypothesis_id\":\"$HID\",\"status\":\"supported\"}"
```
