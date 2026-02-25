import os
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# ייבוא הכלי מהקובץ eda_tools.py
from src.tools.eda_tools import generate_eda_report


class AnalystCrew:
    def __init__(self):
        # הגדרת המודל באמצעות CrewAI LLM class
        self.llm = LLM(
            model="gpt-4o-mini",
            temperature=0.7
        )

    def run(self):
        # 1. הגדרת הסוכן (The Agent)
        analyst = Agent(
            role='Senior Data Analyst',
            goal='Analyze the Amazon dataset, generate visualizations, and provide actionable business insights.',
            backstory="""You are an expert data analyst working for a retail-tech company.
            Your goal is to analyze raw data, create visualizations, and derive meaningful business insights.
            You are precise, analytical, and you always provide clear, actionable recommendations.
            You write your insights in a professional markdown format.""",
            verbose=True,
            memory=True,
            tools=[generate_eda_report],
            llm=self.llm,
            allow_delegation=False
        )

        # 2. הגדרת המשימה (The Task)
        eda_task = Task(
            description="""
            Perform a complete Exploratory Data Analysis on the Amazon products dataset:

            1. Use the 'Generate EDA Report' tool with the file path 'data/amazon.csv'.
            2. The tool will generate an HTML report with 3 charts AND return a statistical summary.
            3. Carefully analyze the statistical summary provided by the tool.
            4. Based on your analysis, write 3-5 actionable business insights in markdown format.

            Your insights should include:
            - Key findings about pricing patterns
            - Category distribution analysis
            - Rating and quality observations
            - Discount strategy recommendations
            - Any correlations or trends you notice

            Format your output as a professional markdown document with headers and bullet points.
            """,
            expected_output="""A markdown document containing:
            1. An executive summary
            2. 3-5 detailed business insights with supporting data
            3. Actionable recommendations for the business
            4. Reference to the generated HTML report path""",
            agent=analyst,
            output_file='insights.md'
        )

        # 3. הגדרת הצוות (The Crew)
        crew = Crew(
            agents=[analyst],
            tasks=[eda_task],
            process=Process.sequential,
            verbose=True
        )

        # הרצת הצוות
        result = crew.kickoff()
        return result


if __name__ == "__main__":
    # בדיקה מקומית אם מריצים את הקובץ ישירות
    my_crew = AnalystCrew()
    result = my_crew.run()
    print(f"########################\nResult: {result}\n########################")
