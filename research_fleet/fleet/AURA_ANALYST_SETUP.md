# Aura Analyst for the research fleet

The research agents run natural-language data analysis over the SingleStore
`portfolio_agents` database through a **real SingleStore Aura Analyst Portal
domain** — never a local NL→SQL substitute (that is a hard rule; a local
`generate_sql` would look like Aura but skip the Portal's crawl/entitlements/
governance, which is the whole point). The agent code (`research_agent/analyst.py`)
calls the managed `/analyst/query` endpoint and, until it is configured,
`analyst.available()` returns False and the agents simply skip the analysis
phase (they do NOT fabricate SQL locally).

Aura Analyst is a **managed Portal service** keyed to a *domain*; it is NOT
provisioned by `python -m factory` or scriptable from this repo. It requires a
one-time Portal action:

## 1. Create + crawl the domain (SingleStore Portal)

1. **Analyst → Create domain**, pointed at this workspace and the
   **`portfolio_agents`** database (workspace `portfolio-agents`, S-2, us-east-1,
   project `2ac411c9-2e3b-4fe6-a910-d631d5d92b5e`).
2. **Crawl it.** Until the crawl finishes there is no schema metadata and data
   questions return prose with `sql: null`. The interesting tables for the agents:
   `research_experiments`, `research_findings`, `research_hypotheses`,
   `research_tasks`, `research_agents`, plus the `prices` panel.
3. **Domain settings → API Keys → Copy Endpoint** — the URL ends `/analyst/chat`;
   the agent derives the `/analyst/query` variant automatically.
4. **Create API Key**, name it (e.g. "research-fleet"), copy it once.

The endpoint URL for this project looks like:

```
https://apps.us-east-1.cloud.singlestore.com/v1/organizations/{orgID}/projects/2ac411c9-2e3b-4fe6-a910-d631d5d92b5e/analyst/chat
```

## 2. Fill in the demo .env

```bash
# demos/portfolio-agents/.env
ANALYST_API_URL=https://apps.us-east-1.cloud.singlestore.com/v1/organizations/{orgID}/projects/2ac411c9-2e3b-4fe6-a910-d631d5d92b5e/analyst/chat
ANALYST_API_KEY=…
```

Smoke-test before trusting it:

```bash
curl -s -X POST "${ANALYST_API_URL%/chat}/query" \
  -H "Authorization: Bearer ${ANALYST_API_KEY}" -H "Content-Type: application/json" \
  -d '{"message":"What is the average sharpe across research_experiments, and how many beat the benchmark?"}' | jq .
```

## 3. Push to the fleet (no code change)

```bash
bash staging/portfolio-agents-prep/fleet/push_aura_key.sh
```

This smoke-tests the endpoint, writes `ANALYST_API_URL`/`ANALYST_API_KEY` into
each node's `/opt/research-agent/.env`, and restarts the agent loop. From then
on every research cycle's ANALYST phase asks Aura a question over SingleStore
(e.g. cross-experiment Sharpe averages, which strategy families beat the
benchmark) and records it in `research_analyst_queries`.

## What the agents ask Aura

Each cycle, after backtesting, the agent poses an NL question over the results it
and the fleet have written — e.g. *"Across research_experiments for strategy_family
'momentum', what is the average sharpe and how many experiments beat the
benchmark?"* Aura returns the generated SQL + executed rows; the agent stores the
question, SQL, row count, and answer in `research_analyst_queries`, and folds the
result into its written finding. This is genuine NL analytics over the fleet's own
accumulating research corpus in SingleStore.
