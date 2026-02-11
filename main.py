import os
from dotenv import load_dotenv

# טעינת משתני סביבה מקובץ .env
load_dotenv()

from src.crews.analyst_crew import AnalystCrew

# בדיקה שהמפתח קיים
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found!")
    print("Please add your API key to the .env file")
    exit(1)

def main():
    print("## Starting the Analyst Crew ##")
    
    # יצירת המופע של הצוות
    analytics_crew = AnalystCrew()
    
    # הרצת הצוות
    result = analytics_crew.run()
    
    print("\n\n########################")
    print("## Task Completed! ##")
    print(f"## Final Result: {result}")
    print("########################")

if __name__ == "__main__":
    main()