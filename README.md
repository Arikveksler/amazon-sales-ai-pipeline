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

## Status

Pipeline infrastructure ready, waiting for Analyst/Scientist crews.

## Team

- **Pipeline Lead:** Arik Veksler
- **Analyst Crew:** TBD
- **Scientist Crew:** TBD
