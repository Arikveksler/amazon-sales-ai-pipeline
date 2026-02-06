# Analyst Crew Package
from .agents import create_analyst_agents
from .tasks import create_analyst_tasks

# TODO: יש להחליף בקוד האמיתי מה-EDA Specialist


def run_analyst_crew(
    input_data_path: str,
    output_dir: str,
    reports_dir: str,
    contracts_dir: str,
) -> dict:
    """
    הרצת Data Analyst Crew

    Crew זה אחראי על:
    - ניקוי הנתונים הגולמיים
    - יצירת EDA report
    - יצירת insights
    - יצירת dataset contract

    Args:
        input_data_path: נתיב לקובץ הנתונים הגולמיים
        output_dir: תיקייה לשמירת נתונים מעובדים
        reports_dir: תיקייה לשמירת דוחות
        contracts_dir: תיקייה לשמירת חוזים

    Returns:
        מילון עם נתיבי כל התוצרים שנוצרו
    """
    # TODO: להחליף ב-CrewAI implementation אמיתי

    raise NotImplementedError(
        "Analyst Crew not implemented yet. "
        "EDA Specialist needs to provide the implementation."
    )
