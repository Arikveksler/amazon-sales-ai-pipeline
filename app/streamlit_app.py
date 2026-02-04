"""
Streamlit App - ממשק משתמש לפרויקט
Streamlit App - User interface for the project

TODO: UI Developer יממש את הממשק המלא
"""

import streamlit as st
from pathlib import Path
import sys

# הוספת נתיב לקוד המקור
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """הפונקציה הראשית של האפליקציה"""

    st.set_page_config(
        page_title="Amazon Sales AI Pipeline",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Amazon Sales AI Pipeline")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Run Pipeline", "View Reports", "About"],
    )

    if page == "Home":
        show_home()
    elif page == "Run Pipeline":
        show_run_pipeline()
    elif page == "View Reports":
        show_reports()
    elif page == "About":
        show_about()


def show_home():
    """דף הבית"""
    st.header("Welcome to Amazon Sales AI Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Analyst Crew")
        st.markdown("""
        - ניקוי נתונים
        - EDA Report
        - תובנות ראשוניות
        - יצירת Dataset Contract
        """)

    with col2:
        st.subheader("🤖 Scientist Crew")
        st.markdown("""
        - Feature Engineering
        - אימון מודל
        - הערכת ביצועים
        - Model Card
        """)

    st.markdown("---")
    st.info("👆 Use the sidebar to navigate between pages")


def show_run_pipeline():
    """דף הרצת Pipeline"""
    st.header("🚀 Run Pipeline")

    st.warning("⚠️ This feature is not yet implemented")

    # Placeholder for file upload
    uploaded_file = st.file_uploader("Upload Amazon Sales CSV", type=["csv"])

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Pipeline", disabled=uploaded_file is None):
        st.info("Pipeline execution will be implemented here")


def show_reports():
    """דף צפייה בדוחות"""
    st.header("📑 View Reports")

    st.warning("⚠️ Reports will appear here after running the pipeline")

    report_type = st.selectbox(
        "Select Report",
        ["EDA Report", "Insights", "Evaluation Report", "Model Card"],
    )

    st.info(f"Selected: {report_type}")


def show_about():
    """דף אודות"""
    st.header("ℹ️ About")

    st.markdown("""
    ## Amazon Sales AI Pipeline

    CrewAI Flow project for analyzing and predicting Amazon sales data.

    ### Architecture
    1. **Raw Data** → Analyst Crew → Clean Data + Contract
    2. Clean Data → **Validation** → Scientist Crew
    3. Scientist Crew → **Model + Reports**

    ### Team
    - Pipeline Lead
    - EDA Specialist
    - ML Specialist
    - UI Developer
    - Business & Docs
    """)


if __name__ == "__main__":
    main()
