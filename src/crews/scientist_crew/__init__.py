# Scientist Crew Package
from .agents import create_scientist_agents
from .tasks import create_scientist_tasks

from crewai import Crew, Process
from pathlib import Path
from loguru import logger
import os


def run_scientist_crew(
    clean_data_path: str,
    contract_path: str,
    features_dir: str,
    models_dir: str,
    reports_dir: str,
) -> dict:
    """
    הרצת Data Scientist Crew.
    Run Data Scientist Crew for ML model training pipeline.

    Crew זה אחראי על:
    - Feature Engineering: Create ML-ready features
    - Model Training: Train Random Forest, XGBoost, Linear Regression with GridSearchCV
    - Model Evaluation: Evaluate and compare models, select best
    - Model Documentation: Create Model Card with responsible AI standards

    Args:
        clean_data_path: Path to clean_data.csv (from Analyst Crew)
        contract_path: Path to dataset_contract.json (from Analyst Crew)
        features_dir: Directory to save engineered features
        models_dir: Directory to save trained model
        reports_dir: Directory to save evaluation reports

    Returns:
        Dictionary with output paths and metrics:
        {
            'features_path': str,
            'model_path': str,
            'evaluation_report_path': str,
            'model_card_path': str,
            'metrics': dict  # Best model test metrics
        }

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If crew execution fails
    """
    logger.info("="*60)
    logger.info("Starting Scientist Crew")
    logger.info("="*60)

    # ========================================================================
    # Step 1: Validate Inputs
    # ========================================================================

    logger.info("Step 1: Validating inputs...")

    # Check input files exist
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(f"Clean data file not found: {clean_data_path}")

    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Contract file not found: {contract_path}")

    logger.info(f"  ✓ Clean data: {clean_data_path}")
    logger.info(f"  ✓ Contract: {contract_path}")

    # ========================================================================
    # Step 2: Create Output Directories
    # ========================================================================

    logger.info("Step 2: Creating output directories...")

    Path(features_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"  ✓ Features dir: {features_dir}")
    logger.info(f"  ✓ Models dir: {models_dir}")
    logger.info(f"  ✓ Reports dir: {reports_dir}")

    # ========================================================================
    # Step 3: Create Agents
    # ========================================================================

    logger.info("Step 3: Creating agents...")

    agents = create_scientist_agents()

    logger.info(f"  ✓ Created {len(agents)} agents:")
    for agent in agents:
        logger.info(f"    - {agent.role}")

    # ========================================================================
    # Step 4: Create Tasks
    # ========================================================================

    logger.info("Step 4: Creating tasks...")

    tasks = create_scientist_tasks(
        agents=agents,
        clean_data_path=clean_data_path,
        contract_path=contract_path,
        features_dir=features_dir,
        models_dir=models_dir,
        reports_dir=reports_dir
    )

    logger.info(f"  ✓ Created {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        logger.info(f"    {i}. {task.agent.role}")

    # ========================================================================
    # Step 5: Create and Run Crew
    # ========================================================================

    logger.info("Step 5: Creating and running crew...")

    scientist_crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,  # Tasks run in order
        verbose=2,  # Maximum verbosity
        memory=False  # No shared memory needed
    )

    logger.info("  ✓ Crew created (sequential process)")
    logger.info("  Starting crew execution...")
    logger.info("  (This may take several minutes due to model training)")
    logger.info("-"*60)

    try:
        result = scientist_crew.kickoff()

        logger.info("-"*60)
        logger.success("✓ Crew execution completed!")

    except Exception as e:
        logger.error(f"✗ Crew execution failed: {e}")
        raise ValueError(f"Scientist Crew execution failed: {e}")

    # ========================================================================
    # Step 6: Validate Outputs
    # ========================================================================

    logger.info("Step 6: Validating outputs...")

    # Define expected output paths
    features_path = os.path.join(features_dir, 'features.csv')
    model_path = os.path.join(models_dir, 'model.pkl')
    eval_report_path = os.path.join(reports_dir, 'evaluation_report.md')
    model_card_path = os.path.join(reports_dir, 'model_card.md')

    # Check all outputs exist
    missing_outputs = []

    if not os.path.exists(features_path):
        missing_outputs.append('features.csv')
    else:
        logger.info(f"  ✓ Features: {features_path}")

    if not os.path.exists(model_path):
        missing_outputs.append('model.pkl')
    else:
        logger.info(f"  ✓ Model: {model_path}")

    if not os.path.exists(eval_report_path):
        missing_outputs.append('evaluation_report.md')
    else:
        logger.info(f"  ✓ Evaluation report: {eval_report_path}")

    if not os.path.exists(model_card_path):
        missing_outputs.append('model_card.md')
    else:
        logger.info(f"  ✓ Model card: {model_card_path}")

    if missing_outputs:
        raise ValueError(f"Missing outputs: {missing_outputs}")

    # ========================================================================
    # Step 7: Extract Metrics
    # ========================================================================

    logger.info("Step 7: Extracting metrics from model...")

    try:
        import joblib
        model_data = joblib.load(model_path)
        metadata = model_data.get('metadata', {})
        test_metrics = metadata.get('test_metrics', {})

        logger.info(f"  ✓ Best model: {metadata.get('model_type', 'Unknown')}")
        logger.info(f"  ✓ Test R²: {test_metrics.get('r2', 0.0):.4f}")
        logger.info(f"  ✓ Test MAE: {test_metrics.get('mae', 0.0):.2f}")

    except Exception as e:
        logger.warning(f"  ⚠ Could not extract metrics: {e}")
        test_metrics = {}

    # ========================================================================
    # Step 8: Return Results
    # ========================================================================

    logger.info("="*60)
    logger.success("Scientist Crew Completed Successfully!")
    logger.info("="*60)

    return {
        'features_path': features_path,
        'model_path': model_path,
        'evaluation_report_path': eval_report_path,
        'model_card_path': model_card_path,
        'metrics': test_metrics
    }
