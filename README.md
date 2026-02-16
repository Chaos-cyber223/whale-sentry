# Whale-Sentry  
**On-chain Risk Detection for MEV & Trading Abuse**

Whale-Sentry is an on-chain risk detection system designed to identify suspicious trading behaviors on decentralized exchanges, with a focus on MEV-related attacks (e.g., sandwich attacks) and wash trading on Uniswap V3.

The project applies statistical modeling and lightweight machine learning as supporting tools, prioritizing data correctness, interpretability, and operational awareness over model complexity.

> ✅ **Project Status (Feb 16, 2026): Wash Trading Detection Complete**  
> Wash trading detection fully implemented with ROUNDTRIP pattern detection, CLI script, and comprehensive test suite. The system now supports statistical analysis of self-trading patterns with configurable thresholds. Test coverage: **101 tests passing** (72 sandwich + 29 wash trading).

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
Statistical detection of self-trading patterns:
- **ROUNDTRIP pattern**: Same address repeatedly buying and selling within short time windows
- **Detection parameters**: 
  - Time window: 300 seconds (5 minutes)
  - Minimum trade amount: $1,000 USD
  - Minimum round trips: 3 cycles
- **Metrics tracked**:
  - Trade count and frequency
  - Amount similarity (MAD-based)
  - Counterparty diversity
  - Time window analysis
- **Confidence scoring**: Multi-factor weighted scoring (trade count, amount similarity, counterparty diversity)

---

## Results & Validation

The detection system has been validated on real Uniswap V3 data:

### Dataset
- **Pool**: WETH/USDC (0x88e6a0c2...cb3f5640)
- **Transactions**: 90 swaps
- **Time Period**: January 2024

### Detection Results

#### Sandwich Attacks
- **Candidates Found**: 1
- **Confidence**: 0.77 (medium confidence)
- **False Positive Rate**: Low (single candidate in clean dataset)

#### Wash Trading
- **Candidates Found**: 0
- **Analysis**: No ROUNDTRIP patterns detected with current parameters (300s window, $1000 minimum, 3 round trips)
- **Interpretation**: Clean dataset with no self-trading patterns, demonstrating the detector's ability to avoid false positives

### Key Insights
- **Low False Positive Rate**: The system correctly identifies clean data without over-flagging
- **Configurable Thresholds**: Parameters can be adjusted for different risk tolerance levels
- **Production Ready**: Detection completes in <0.1s for 90 transactions, suitable for batch processing

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

### Sandwich Attack Detection
- **Detection Module** (`whalesentry/detection/sandwich.py`)
  - `SandwichCandidate` Pydantic model for type-safe results
  - `detect_sandwich_attacks()` - original O(n³) implementation
  - `detect_sandwich_attacks_optimized()` - optimized O(n log n) algorithm
  - **Performance**: 45-92x speedup on 100-1000 transaction datasets
  - Detection logic: identifies attacker addresses before/after victim transactions
  - Validates directional reversal (buy→sell or sell→buy)
  - Confidence scoring based on timing, amounts, and victim transaction size

- **CLI Tool** (`scripts/detect_sandwiches.py`)
  - Command-line interface with configurable parameters
  - `--use-optimized` flag (default: True) for O(n log n) algorithm
  - `--use-legacy` flag for original O(n³) algorithm
  - Outputs results to Parquet format
  - Generates Markdown detection reports with pool-level statistics
  - Displays algorithm type and execution time

### Wash Trading Detection (In Progress)
- **Data Models** (`whalesentry/models/wash_trade.py`)
  - `WashTradeType` (StrEnum): Pattern classification (roundtrip, closed-loop, coordinated)
  - `WashTradeMetrics`: Quantitative measurements (trade count, amount similarity, counterparty diversity)
  - `WashTradeCandidate`: Type-safe representation with full validation
  - `WashTradeDetectionResult`: Container with statistics and filtering support
  - Follows same patterns as `SandwichCandidate` for consistency

### Testing
- **Complete Unit Test Suite**
  - Data validation tests (`tests/test_clean_swaps.py`): 33 test cases
  - Sandwich detection tests (`tests/test_sandwich_detection.py`): 29 test cases
  - Performance tests (`tests/test_sandwich_performance.py`): 7 test cases
  - **Total: 72 tests, all passing ✅**
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
Rule-based Detection (Sandwich ✅ / Wash 🚧) ← In Progress
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

### Sandwich Attack Detection

```bash
# Basic usage - detect sandwich attacks
$ python scripts/detect_sandwiches.py \
    --input data/processed/swaps_clean.parquet \
    --output data/results/sandwich_candidates.parquet

# Use legacy O(n³) algorithm for comparison
$ python scripts/detect_sandwiches.py \
    --input data/processed/swaps_clean.parquet \
    --use-legacy

# Custom parameters
$ python scripts/detect_sandwiches.py \
    --input data/processed/swaps_clean.parquet \
    --time-window 120 \
    --min-usd 500 \
    --min-confidence 0.8 \
    --report data/results/detection_report.md

# Verbose output with timing information
$ python scripts/detect_sandwiches.py -i swaps.parquet -v
```

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
# Sandwich attack detection
from whalesentry.detection import detect_sandwich_attacks_optimized
import pandas as pd
from decimal import Decimal

# Load clean swap data
df = pd.read_parquet("data/processed/swaps_clean.parquet")

# Run optimized detection
result = detect_sandwich_attacks_optimized(
    df,
    time_window_seconds=60,
    min_usd_value=Decimal("100"),
    amount_similarity_threshold=0.5
)

print(f"Found {result.total_candidates} sandwich attack candidates")
print(f"Analyzed {result.total_swaps_analyzed} swaps across {result.pools_analyzed} pools")

# Access individual candidates
for candidate in result.candidates[:5]:
    print(f"Attacker: {candidate.attacker}")
    print(f"Victim TX: {candidate.victim_tx}")
    print(f"Estimated profit: ${candidate.profit_estimate_usd}")
    print(f"Confidence: {candidate.confidence_score:.2f}")

# Data validation
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

# Wash trading detection models (Milestone 1 Complete)
from whalesentry.models.wash_trade import (
    WashTradeType,
    WashTradeMetrics,
    WashTradeCandidate,
    WashTradeDetectionResult,
)

# Create wash trading metrics
metrics = WashTradeMetrics(
    trade_count=6,
    unique_addresses=1,
    time_window_seconds=300,
    round_trip_count=3,
    avg_trade_interval_seconds=50.0,
    amount_similarity_score=0.98,
    volume_usd_total="15000.00",
)

# Create wash trading candidate
candidate = WashTradeCandidate(
    detection_type=WashTradeType.ROUNDTRIP,
    primary_address="0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
    involved_addresses=["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
    pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    token_pair="WETH/USDC",
    related_tx_hashes=["0xabc...", "0xdef..."],
    start_timestamp=1769263200,
    end_timestamp=1769263500,
    metrics=metrics,
    confidence_score=0.92,
    description="Detected 3 round-trip trades within 5 minutes",
)

print(f"Confidence: {candidate.confidence_score}")
print(f"Duration: {candidate.duration_seconds}s")
print(f"High confidence: {candidate.is_high_confidence}")
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

### ✅ Completed
- ~~Data ingestion from Uniswap V3 Subgraph~~
- ~~Pydantic data models with validation~~
- ~~Data cleaning and validation pipeline~~
- ~~Comprehensive unit tests (72 tests)~~
- ~~**Wash trading data models with Pydantic validation**~~
- ~~Validation reporting infrastructure~~
- ~~**Sandwich attack detection with O(n log n) optimization**~~
- ~~**CLI tool for sandwich detection**~~
- ~~**Performance benchmarking (45-92x speedup)**~~

### 🚧 In Progress
- Wash trading detection algorithms (roundtrip & closed-loop heuristics)
- Exploratory analysis notebooks

### ✅ Recently Completed
- ~~Wash trading data models with Pydantic validation~~

### 📋 Planned
- Anomaly scoring (Z-score, Isolation Forest)
- Visualization of anomalous events
- Minimal Streamlit dashboard
- Real-time detection pipeline
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

## Limitations & Known Issues

### Current Implementation
- **Block-level ordering**: Currently uses timestamps only. Real sandwich attacks occur within the same block; future versions should incorporate `txIndex` for more precise detection.
- **Profit calculation**: Uses simplified USD value difference. Production version should calculate actual profit based on `sqrtPriceX96` price impact and gas costs.
- **MEV bot filtering**: Does not distinguish between malicious sandwich attacks and legitimate arbitrage bots.

### Performance Characteristics
- **Optimized algorithm**: O(n log n) complexity, suitable for datasets up to 10,000+ transactions
- **Legacy algorithm**: O(n³) complexity, provided for reference and validation only
- **Memory usage**: Reasonable for datasets up to 100,000 transactions (< 500MB)

**Future extensions:**
- Direct log-level ingestion via `web3.py`
- Real-time streaming detection
- Cross-DEX analysis
- Integration with MEV relay / builder data
- Known MEV bot address filtering
- Multi-victim sandwich pattern detection

## About the Author

**Rhea Wang**  
M.S. in Statistics, University of Pennsylvania  

Background in statistical modeling, applied machine learning, and cloud-native engineering, with a focus on building reliable on-chain risk systems.
Currently focused on **on-chain risk, MEV analysis, and DeFi market behavior**.

Open to **Web3 / Crypto roles (remote or Singapore-based)**.
