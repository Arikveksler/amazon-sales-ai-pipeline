# Scientist Crew Package
from .agents import create_scientist_agents
from .tasks import create_scientist_tasks

# TODO: יש להחליף בקוד האמיתי מה-ML Specialist


def run_scientist_crew(
    clean_data_path: str,
    contract_path: str,
    features_dir: str,
    models_dir: str,
    reports_dir: str,
) -> dict:
    """
    הרצת Data Scientist Crew

    Crew זה אחראי על:
    - Feature Engineering
    - אימון מודל
    - Evaluation
    - יצירת Model Card

    Args:
        clean_data_path: נתיב לנתונים הנקיים
        contract_path: נתיב לחוזה הנתונים
        features_dir: תיקייה לשמירת features
        models_dir: תיקייה לשמירת מודלים
        reports_dir: תיקייה לשמירת דוחות

    Returns:
        מילון עם נתיבי כל התוצרים שנוצרו
    """
    # TODO: להחליף ב-CrewAI implementation אמיתי

    raise NotImplementedError(
        "Scientist Crew not implemented yet. "
        "ML Specialist needs to provide the implementation."
    )
