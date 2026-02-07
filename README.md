# Whale-Sentry  
**On-chain Risk Detection for MEV & Trading Abuse**

Whale-Sentry is an on-chain risk detection system designed to identify suspicious trading behaviors on decentralized exchanges, with a focus on MEV-related attacks (e.g., sandwich attacks) and wash trading on Uniswap V3.

The project applies statistical modeling and lightweight machine learning as supporting tools, prioritizing data correctness, interpretability, and operational awareness over model complexity.

> ✅ **Project Status (Feb 7, 2026): Data Infrastructure Complete**  
> Core data pipeline and validation infrastructure is now production-ready. Pydantic models, comprehensive validation tools, and full test coverage are in place. Detection algorithms in development.

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

## Completed Features ✅

### Data Infrastructure
- **Pydantic Data Models** (`whalesentry/models/`)
  - Type-safe `SwapEvent` model with comprehensive validation
  - Ethereum address format validation (regex-based)
  - Transaction hash validation
  - Tick range validation (Uniswap V3 bounds)
  - Decimal amount validation for precision
  - `SwapDataFrame` container for batch operations

### Validation & Quality Assurance
- **Validation Tools** (`whalesentry/validation/`)
  - `SwapValidator` class for DataFrame validation
  - Detailed error reporting with row-level tracking
  - Duplicate detection and statistics
  - Schema validation against required columns
  - Amount consistency checks (opposite signs)
  - USD value validation
  - `CleaningReport` for markdown report generation

### Data Processing
- **Improved Cleaning Pipeline** (`scripts/clean_swaps.py`)
  - Command-line interface with argparse
  - Pydantic-based validation integration
  - Configurable input/output paths
  - USD value filtering support
  - Comprehensive logging with structured format
  - Strict mode for CI/CD integration
  - Automatic report generation

### Testing
- **Complete Unit Test Suite** (`tests/test_clean_swaps.py`)
  - Pydantic model validation tests (13 test cases)
  - SwapDataFrame collection tests (6 test cases)
  - Validator functionality tests (7 test cases)
  - Cleaning report tests (3 test cases)
  - Data cleaning pipeline tests (4 test cases)
  - Integration tests including parquet roundtrip

---

## Project Architecture

```text
On-chain Data (Uniswap V3)
        ↓
Data Ingestion (GraphQL)
        ↓
Cleaning & Feature Engineering ← ✅ COMPLETE
        ↓
Rule-based Detection (Sandwich / Wash) ← In Progress
        ↓
Anomaly Scoring (Stat / ML) ← Planned
        ↓
Analysis Notebooks + Risk Dashboard ← Planned
```

## Technology Stack

- **Python** (data pipelines, modeling)
- **Pydantic** (type-safe data validation)
- **pandas / NumPy / scikit-learn**
- **DuckDB / Parquet** (analytical storage)
- **pytest** (comprehensive testing)
- **Jupyter Notebooks** (research & reporting)
- **Streamlit** (lightweight risk dashboard)

---

## Usage Examples

### Data Cleaning

```bash
# Basic usage - clean and validate swap data
$ python scripts/clean_swaps.py --input data/raw/swaps.parquet

# Custom output path with validation report
$ python scripts/clean_swaps.py \
    --input data/raw/swaps.parquet \
    --output data/processed/swaps_clean.parquet \
    --report data/processed/cleaning_report.md

# Filter by minimum USD value (e.g., transactions > $1000)
$ python scripts/clean_swaps.py \
    --input data/raw/swaps.parquet \
    --min-usd 1000 \
    --output data/processed/large_swaps.parquet

# Strict mode - fail on any validation error (for CI/CD)
$ python scripts/clean_swaps.py --input swaps.parquet --strict

# Verbose logging for debugging
$ python scripts/clean_swaps.py -i swaps.parquet -v
```

### Programmatic Usage

```python
from whalesentry.models.swap import SwapEvent, SwapDataFrame
from whalesentry.validation import SwapValidator, CleaningReport
import pandas as pd

# Load and validate data
df = pd.read_parquet("data/raw/swaps.parquet")
validator = SwapValidator()
result = validator.validate_dataframe(df)

print(f"Validation rate: {result.success_rate:.1f}%")
print(f"Valid records: {result.valid_records}/{result.total_records}")

# Generate report
report = CleaningReport()
report.add_validation_result(result)
report.add_statistics(df)
markdown = report.generate_markdown()
```

### Running Tests

```bash
# Run all tests
$ pytest tests/ -v

# Run with coverage
$ pytest tests/ --cov=whalesentry --cov-report=html

# Run specific test file
$ pytest tests/test_clean_swaps.py -v
```

---

## Development Roadmap

### ✅ Completed (Week 1)
- ~~Data ingestion from Uniswap V3 Subgraph~~
- ~~Pydantic data models with validation~~
- ~~Data cleaning and validation pipeline~~
- ~~Comprehensive unit tests~~
- ~~Validation reporting infrastructure~~

### 🚧 In Progress (Week 2)
- Initial sandwich attack rule-based detection
- Wash trading detection heuristics
- Exploratory analysis notebooks

### 📋 Planned
- Anomaly scoring (Z-score, Isolation Forest)
- Visualization of anomalous events
- Minimal Streamlit dashboard
- Documentation & result examples

---

## Limitations & Future Work

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
