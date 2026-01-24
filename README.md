# whale-sentry
On-chain anomaly detection for MEV sandwich & wash trading on Uniswap V3.

## MVP Architecture (current)
```mermaid
flowchart TD
  A[Uniswap V3 Subgraph] --> B[Ingestion: scripts/fetch_swaps.py]
  B --> C[Raw Parquet: data/raw]
  C --> D[Cleaning: scripts/clean_swaps.py]
  D --> E[Clean Parquet: data/clean]
  E --> F[Detectors (next): sandwich / wash]
  F --> G[Scoring + Dashboard (next)]
```

## Quickstart
1. Create & activate a Python 3.11 virtualenv:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install the package (editable) with runtime + dev dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Configure your subgraph endpoint and optional API key:
   ```bash
   cp configs/.env.example .env
   # edit .env and set SUBGRAPH_ENDPOINT (+ SUBGRAPH_API_KEY when required)
   ```
4. Fetch recent swaps for a pool (example uses 24h window):
   ```bash
   python scripts/fetch_swaps.py --pool <POOL_ADDRESS> --hours 24
   ```
5. Clean the raw swaps for downstream detectors:
   ```bash
   python scripts/clean_swaps.py
   ```

## Limitations & Notes
- Subgraph schemas vary (different field casing or nesting). The ingestion layer normalizes common fields and leaves missing attributes as nulls; review logs if columns are absent.
- Public subgraphs enforce rate limits; the SubgraphClient retries with exponential backoff but heavy backfills may still need manual throttling.
- `SUBGRAPH_ENDPOINT` is required; scripts exit early with instructions if it is missing.
- Hosted providers that require auth tokens (e.g., Gateway endpoints) need `SUBGRAPH_API_KEY`, which the fetch script converts into an `Authorization: Bearer <token>` header.

## Next Steps
- Extend ingestion beyond Uniswap V3 subgraph (direct RPC + traces) and persist to DuckDB/warehouse.
- Build sandwich & wash-trading detectors consuming the clean parquet outputs.
- Add CI (lint + pytest) and containerized runtime for scheduled jobs.
