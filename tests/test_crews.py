"""
בדיקות עבור Crews
Tests for Crews modules

הערה: בדיקות אלו הן placeholders שיעודכנו כאשר ה-Crews יממשו
Note: These tests are placeholders to be updated when Crews are implemented
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnalystCrew:
    """בדיקות עבור Analyst Crew"""

    def test_analyst_crew_not_implemented(self):
        """
        בדיקה: Analyst Crew עדיין לא מומש
        Test: Analyst Crew raises NotImplementedError
        """
        from src.crews.analyst_crew import run_analyst_crew

        with pytest.raises(NotImplementedError):
            run_analyst_crew(
                input_data_path="test.csv",
                output_dir="output",
                reports_dir="reports",
                contracts_dir="contracts",
            )

    def test_analyst_agents_not_implemented(self):
        """
        בדיקה: Analyst agents עדיין לא מומשו
        Test: Analyst agents raise NotImplementedError
        """
        from src.crews.analyst_crew.agents import create_analyst_agents

        with pytest.raises(NotImplementedError):
            create_analyst_agents()

    def test_analyst_tasks_not_implemented(self):
        """
        בדיקה: Analyst tasks עדיין לא מומשו
        Test: Analyst tasks raise NotImplementedError
        """
        from src.crews.analyst_crew.tasks import create_analyst_tasks

        with pytest.raises(NotImplementedError):
            create_analyst_tasks()


class TestScientistCrew:
    """בדיקות עבור Scientist Crew"""

    def test_scientist_crew_not_implemented(self):
        """
        בדיקה: Scientist Crew עדיין לא מומש
        Test: Scientist Crew raises NotImplementedError
        """
        from src.crews.scientist_crew import run_scientist_crew

        with pytest.raises(NotImplementedError):
            run_scientist_crew(
                clean_data_path="clean.csv",
                contract_path="contract.json",
                features_dir="features",
                models_dir="models",
                reports_dir="reports",
            )

    def test_scientist_agents_not_implemented(self):
        """
        בדיקה: Scientist agents עדיין לא מומשו
        Test: Scientist agents raise NotImplementedError
        """
        from src.crews.scientist_crew.agents import create_scientist_agents

        with pytest.raises(NotImplementedError):
            create_scientist_agents()

    def test_scientist_tasks_not_implemented(self):
        """
        בדיקה: Scientist tasks עדיין לא מומשו
        Test: Scientist tasks raise NotImplementedError
        """
        from src.crews.scientist_crew.tasks import create_scientist_tasks

        with pytest.raises(NotImplementedError):
            create_scientist_tasks()


# הרצת הבדיקות
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
