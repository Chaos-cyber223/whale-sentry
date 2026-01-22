# Whale-Sentry  
**On-chain Risk Detection for MEV & Trading Abuse**

Whale-Sentry is an on-chain risk detection system designed to identify suspicious trading behaviors on decentralized exchanges, with a focus on MEV-related attacks (e.g., sandwich attacks) and wash trading on Uniswap V3.

The project applies statistical modeling and lightweight machine learning as supporting tools, prioritizing data correctness, interpretability, and operational awareness over model complexity.

> 🚧 **Project Status (Jan 22, 2026): Actively under development**  
> Initial MVP is planned within **2 weeks**, focusing on reproducible data pipelines and interpretable risk signals.

---

## Motivation

Decentralized finance introduces transparency at the data level, but **not at the behavior level**.  
Malicious or exploitative trading strategies—such as sandwich attacks and wash trading—are often hidden within large volumes of legitimate transactions.

Whale-Sentry is built to answer a practical question:

> **Can we systematically detect suspicious on-chain trading behavior using statistical signals and machine learning, rather than heuristics alone?**

This project is particularly relevant for:
- DeFi protocol risk monitoring
- MEV analysis and mitigation research
- On-chain compliance, fraud detection, and market surveillance

---

## Core Focus

- **On-chain data analysis** (Uniswap V3 swaps & liquidity events)
- **MEV-aware anomaly detection**
- **Explainable risk signals**, not black-box predictions
- Research-oriented engineering with production-minded constraints

---

## Data Sources

- **Uniswap V3 Subgraph**
  - Swap events (transaction behavior)
  - Liquidity events (Mint / Burn)
- Initial scope:
  - Ethereum mainnet
  - Major liquid pools (e.g., WETH / USDC)

All data pipelines are designed to be **fully reproducible**.

---

## Detection Targets

### 1. Sandwich Attacks (MEV)
Pattern-based detection combined with anomaly scoring:
- Same-pool, short-interval transaction sequences
- Attacker address appearing before and after a victim trade
- Directional reversal consistent with price manipulation
- Abnormal price or volume impact around victim transactions

### 2. Wash Trading
Behavioral and statistical signals, including:
- High-frequency round-trip trading
- Low counterparty diversity
- Repetitive trade sizes
- Extremely short inter-trade intervals

---

## Methodology

The system follows a two-layer approach:

1. **Rule-based candidate generation**
   - Interpretable, MEV-aware heuristics for high recall
2. **Statistical & ML-based scoring**
   - Robust Z-score (MAD-based)
   - Isolation Forest for anomaly ranking

This design prioritizes **explainability first**, then **model-based prioritization**.
This design explicitly avoids black-box decision making and favors signals that can be audited, reasoned about, and operationally monitored.

---

## Project Architecture (Planned)

```text
On-chain Data (Uniswap V3)
        ↓
Data Ingestion (GraphQL)
        ↓
Cleaning & Feature Engineering
        ↓
Rule-based Detection (Sandwich / Wash)
        ↓
Anomaly Scoring (Stat / ML)
        ↓
Analysis Notebooks + Risk Dashboard

```

## Technology Stack

- **Python** (data pipelines, modeling)
- **pandas / NumPy / scikit-learn**
- **DuckDB / Parquet** (analytical storage)
- **Jupyter Notebooks** (research & reporting)
- **Streamlit** (lightweight risk dashboard)


## Development Roadmap (MVP – 2 Weeks)

### Week 1
- Data ingestion from Uniswap V3 Subgraph
- Data cleaning and feature engineering
- Initial sandwich attack rule-based detection
- Exploratory analysis notebooks

### Week 2
- Wash trading detection
- Anomaly scoring (Z-score, Isolation Forest)
- Visualization of anomalous events
- Minimal Streamlit dashboard
- Documentation & result examples

## Limitations & Future Work

- Precise block-level transaction ordering (txIndex) is not fully integrated in the MVP
- Profit attribution for MEV actors is approximated, not exact
- These limitations are explicitly documented to avoid overconfidence in risk signals produced by the system.

**Future extensions:**
- Direct log-level ingestion via `web3.py`
- Real-time streaming detection
- Cross-DEX analysis
- Integration with MEV relay / builder data

## About the Author

**Rhea Wang**  
M.S. in Statistics, University of Pennsylvania  

Background in statistical modeling, applied machine learning, and cloud-native engineering, with a focus on building reliable on-chain risk systems.
Currently focused on **on-chain risk, MEV analysis, and DeFi market behavior**.

Open to **Web3 / Crypto roles (remote or Singapore-based)**.
