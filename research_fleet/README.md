# Research Agent Fleet — OpenClaw × NemoClaw × SingleStore

A fleet of **5 autonomous auto-research agents**, each running **OpenClaw through
NVIDIA NemoClaw** (inside an NVIDIA OpenShell sandbox) on its own EC2, that
research and test S&P 500 trading strategies and write their
hypotheses / experiments / findings to **SingleStore over time** — with Qwen
`VECTOR(1024)` semantic recall so each agent builds on the whole fleet's prior
findings. Data analysis over SingleStore runs through a **real Aura Analyst
Portal domain** (never a local NL→SQL substitute).

Extends the `portfolio-agents` demo; shares its SingleStore workspace + the
`portfolio_agents` database (and the 1.9M-row `prices` panel the agents backtest
against).

## How inference works (the key integration)

NemoClaw's built-in Bedrock adapter is hard-gated to real AWS hostnames, so it
can't hit the code-factory's SingleStore-hosted Anthropic gateway directly. The
bridge is a tiny **OpenAI-compatible → Bedrock-Converse shim** on each host:

```
OpenClaw (in OpenShell sandbox)
  → inference.local              (OpenShell managed forward; injects the JWT)
  → http://host.openshell.internal:11500/v1   (host inference shim)
  → SingleStore Bedrock Converse (unsigned + Authorization: Bearer <JWT>)
  → Claude (haiku/sonnet/opus)
```

Credential isolation is preserved: the sandbox only ever sees `inference.local`;
the raw JWT never enters the sandbox. NemoClaw is onboarded as
`provider=custom`, `preferred-api=openai-completions`, endpoint the shim.

## Architecture

```
5x EC2 t3.xlarge (Ubuntu 24.04, same VPC/subnet as the L4 GPU box, us-east-1c)
each node:
  ├ inference-shim.service   OpenAI→Bedrock shim :11500  (fleet/inference_shim.py)
  ├ NemoClaw + OpenShell     docker sandbox running OpenClaw v2026.7.1
  └ research-agent.service   the loop (research_agent/agent_loop.py):
        claim task ─▶ recall prior findings (VECTOR <*>) ─▶ hypothesize (Claude)
        ─▶ backtest over SingleStore prices ─▶ (Aura Analyst NL analysis)
        ─▶ write finding (embed + persist) ─▶ mark done ─▶ loop
                              │
                              ▼
             SingleStore  portfolio_agents DB  (results accumulate over time)
      research_agents / research_tasks / research_hypotheses /
      research_experiments / research_findings / research_activity /
      research_analyst_queries
```

## Components

- `research_agent/` — the per-node runtime:
  - `research_db.py` — SingleStore API: atomic `claim_task` (guarded UPDATE so N
    agents don't collide), register/heartbeat, log_activity, add_hypothesis,
    record_experiment, write_finding (+Qwen embed), `recall_findings`
    (`embedding <*> q` DOT_PRODUCT across the whole fleet).
  - `backtest.py` — NumPy/pandas backtester over the sp500 `prices` table;
    families: equal_weight, momentum, mean_reversion, vol_target, low_vol/factor,
    risk_parity, regime — all vs a 1/N benchmark.
  - `llm_driver.py` — Claude via the code-factory Bedrock-Converse+Bearer path
    (structured hypothesis JSON + finding text).
  - `analyst.py` — Aura Analyst `/analyst/query` client; `available()` gates on
    `ANALYST_API_URL`/`ANALYST_API_KEY` and SKIPS if unset (never substitutes).
  - `agent_loop.py` — the claim→recall→hypothesize→backtest→analyze→finding loop.
- `fleet/` — deployment:
  - `inference_shim.py` — the OpenAI→Bedrock shim.
  - `userdata_base.sh` — thin EC2 userdata (docker + node22 + python).
  - `provision_node.sh` — post-boot SSH provisioning (ship pkg, shim service,
    NemoClaw install, agent service).
  - `onboard_openshell_remote.sh` — the VERIFIED non-interactive OpenShell +
    OpenClaw onboard (custom provider → shim). Run per node.
  - `launch_fleet.sh` — boot N t3.xlarge nodes in the GPU VPC.
  - `push_aura_key.sh` + `AURA_ANALYST_SETUP.md` — wire a real Aura Analyst domain.
- `research_schema.sql` / `seed_research_tasks.py` — schema + work-queue seeding.

## Operate

```bash
source /Users/billscolinos/Documents/code_factory/.venv/bin/activate
# add more research briefs to the shared queue:
python seed_research_tasks.py       # (idempotent; agents claim atomically)

# inspect fleet output:
python - <<'PY'
import sys; sys.path.insert(0,'.'); from research_agent import research_db as rdb
for r in rdb.query("SELECT agent_id,status,heartbeat_at FROM research_agents"): print(r)
print("findings:", rdb.query("SELECT COUNT(*) c FROM research_findings")[0]['c'])
PY
```

`.env` is read from the demo root (`../.env`) or `/opt/research-agent/.env` on a node.

## Live infra (as built 2026-08-10)

5x t3.xlarge, tag `fleet=research-agents`, VPC `vpc-282c2551` / subnet
`subnet-2e70a84a` / us-east-1c, SG `sg-0095be14b8bf0ed08` (SSH from operator /32 +
intra-VPC), key `staging/research-fleet-key.pem`:

| agent | focus | instance | public IP |
|---|---|---|---|
| research-01 | momentum | i-0d800c86f3ee2c925 | 44.222.71.169 |
| research-02 | regime | i-007a30f3b0a95f80d | 100.57.169.26 |
| research-03 | mean_reversion | i-0dc561afb6c2dfee8 | 3.239.70.218 |
| research-04 | vol_target | i-0a2977dc227921c06 | 100.54.138.145 |
| research-05 | factor | i-0e463ea0c39092f39 | 44.204.255.4 |

All 5 run the genuine OpenShell sandbox (OpenClaw v2026.7.1, healthy) + the
research loop. Public IPs drift on stop/start — re-resolve via
`aws ec2 describe-instances --filters Name=tag:fleet,Values=research-agents`.

## Teardown (stops billing — 5x t3.xlarge ≈ $0.83/hr)

```bash
AWS_SHARED_CREDENTIALS_FILE=~/.aws/credentials_cf AWS_PROFILE=cf_gpu \
  aws ec2 terminate-instances --region us-east-1 --instance-ids \
  i-0d800c86f3ee2c925 i-007a30f3b0a95f80d i-0dc561afb6c2dfee8 \
  i-0a2977dc227921c06 i-0e463ea0c39092f39
```

Findings persist in SingleStore after teardown.
