# Amazon Sales AI Pipeline

CrewAI-based pipeline for analyzing Amazon India sales data.
Analyst Crew cleans the raw data, Scientist Crew builds an ML model and generates evaluation reports.

## Project Structure

```
amazon-sales-ai-pipeline/
├── src/
│   ├── crews/
│   │   ├── analyst_crew/    # Data cleaning agent, tools & tasks
│   │   └── scientist_crew/  # ML modeling agent (TBD)
│   ├── flow/
│   │   ├── main_flow.py     # Pipeline orchestrator
│   │   ├── validators.py    # Validation between steps
│   │   └── state_manager.py # JSON-based run state
│   └── utils/
│       └── error_handler.py # Custom exceptions, retry, logging
├── data/
│   ├── raw/                 # Raw CSV files
│   ├── processed/           # Cleaned data
│   └── contracts/           # Dataset contract JSON
├── outputs/
│   ├── models/              # Trained model (.pkl)
│   └── reports/             # Evaluation & model card
├── logs/                    # Pipeline logs
└── tests/
```

## Installation

```bash
git clone https://github.com/Arikveksler/amazon-sales-ai-pipeline.git
cd amazon-sales-ai-pipeline
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

```bash
python src/flow/main_flow.py
```

## Week 2: Feature Engineering

Transforms the raw Amazon dataset into a numerical `features.csv` file (12 columns, 1465 rows) ready for ML model training.

**Run the script:**
```bash
python src/tools/feature_engineering.py
```

**Run the tests:**
```bash
python -m unittest tests.test_feature_engineering -v
```

**Output:** `data/features/features.csv`

For a full description of every feature column (what it represents, how it's calculated, and its data type), see [docs/FEATURE_REGISTRY.md](docs/FEATURE_REGISTRY.md).

## Status

Pipeline infrastructure ready. Analyst Crew (Week 1) and Feature Engineering (Week 2) complete.

## Team

- **Pipeline Lead:** Arik Veksler
- **Analyst Crew:** TBD
- **Scientist Crew:** TBD
