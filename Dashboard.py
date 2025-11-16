import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #ffd93d;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #6bcf7f;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Hardcoded Login System
def check_login(username, password):
    valid_users = {
        "admin": "admin123",
        "teacher": "teacher123",
        "researcher": "research123"
    }
    return valid_users.get(username) == password


# Login Page
def login_page():
    st.markdown('<div class="main-header">🎓 Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")


# Feature Engineering Function
def perform_feature_engineering(df):
    """Perform feature engineering on the uploaded dataset"""

    # Create average test score
    test_score_cols = [col for col in ['TestScore_Math', 'TestScore_Reading', 'TestScore_Science'] if col in df.columns]
    if test_score_cols:
        df['AvgTestScore'] = df[test_score_cols].mean(axis=1)

    # Create performance categories based on GPA
    if 'GPA' in df.columns:
        def categorize_performance(gpa):
            if gpa >= 3.5:
                return 'High_Achiever'
            elif gpa >= 2.5:
                return 'Average_Performer'
            else:
                return 'Struggling_Learner'

        df['PerformanceCategory'] = df['GPA'].apply(categorize_performance)

    # Create engagement score
    if 'AttendanceRate' in df.columns and 'StudyHours' in df.columns:
        df['EngagementScore'] = ((df['AttendanceRate'] * 0.6) + (df['StudyHours'] / 4 * 0.4)) * 100

    # Create support index
    if 'ParentalEducation' in df.columns and 'ParentSupport' in df.columns:
        education_map = {'<HS': 1, 'HS': 2, 'SomeCollege': 3, 'Bachelors+': 4}
        df['ParentalEducation_Numeric'] = df['ParentalEducation'].map(education_map)
        df['SupportIndex'] = ((df['ParentalEducation_Numeric'] / 4 * 0.5) + (df['ParentSupport'] * 0.5)) * 100

    return df


# Train Random Forest Models
def train_models(df):
    """Train Random Forest models for performance and risk prediction"""

    # Prepare features
    feature_candidates = ['Age', 'SES_Quartile', 'TestScore_Math', 'TestScore_Reading',
                          'TestScore_Science', 'AvgTestScore', 'AttendanceRate', 'StudyHours',
                          'EngagementScore', 'SupportIndex', 'PartTimeJob', 'ParentSupport', 'FreeTime']

    available_features = [f for f in feature_candidates if f in df.columns]

    # Add encoded categorical variables
    if 'Gender' in df.columns:
        df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
        available_features.append('Gender_Encoded')

    if 'SchoolType' in df.columns:
        df['SchoolType_Encoded'] = df['SchoolType'].map({'Private': 1, 'Public': 0})
        available_features.append('SchoolType_Encoded')

    # Performance Classification Model
    if 'PerformanceCategory' in df.columns:
        df['Performance_Encoded'] = df['PerformanceCategory'].map({
            'High_Achiever': 0, 'Average_Performer': 1, 'Struggling_Learner': 2
        })

        X = df[available_features]
        y_perf = df['Performance_Encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y_perf, test_size=0.2, random_state=42)

        rf_performance = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_performance.fit(X_train, y_train)

        perf_accuracy = accuracy_score(y_test, rf_performance.predict(X_test))
        df['Predicted_Performance'] = rf_performance.predict(X)
        df['Performance_Probability'] = rf_performance.predict_proba(X)[:, 2]  # Probability of struggling

    else:
        rf_performance = None
        perf_accuracy = None

    # Risk Prediction Model
    if 'PerformanceCategory' in df.columns:
        df['HighRisk'] = (df['PerformanceCategory'] == 'Struggling_Learner').astype(int)
        y_risk = df['HighRisk']

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42)

        rf_risk = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_risk.fit(X_train_r, y_train_r)

        risk_accuracy = accuracy_score(y_test_r, rf_risk.predict(X_test_r))
        df['Predicted_Risk_Probability'] = rf_risk.predict_proba(X)[:, 1]
        df['Predicted_HighRisk'] = rf_risk.predict(X)

    else:
        rf_risk = None
        risk_accuracy = None

    return df, rf_performance, rf_risk, perf_accuracy, risk_accuracy, available_features


# Main Dashboard
def main_dashboard():
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Welcome, {st.session_state.username}!")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # File upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")

            # Show basic info
            st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            # Perform feature engineering
            with st.spinner("Performing feature engineering..."):
                df_processed = perform_feature_engineering(df)

            # Train models
            with st.spinner("Training Random Forest models..."):
                df_final, rf_perf, rf_risk, perf_acc, risk_acc, features = train_models(df_processed)

            # Store in session state
            st.session_state.df = df_final
            st.session_state.models_trained = True
            st.session_state.perf_accuracy = perf_acc
            st.session_state.risk_accuracy = risk_acc
            st.session_state.features = features

        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            return

    # Check if data is loaded
    if 'df' not in st.session_state:
        st.info("Please upload a CSV file to begin analysis")
        return

    df = st.session_state.df

    # Main dashboard content
    st.markdown('<div class="main-header">📊 Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_students = len(df)
        st.metric("Total Students", f"{total_students:,}")

    with col2:
        if 'PerformanceCategory' in df.columns:
            struggling = (df['PerformanceCategory'] == 'Struggling_Learner').sum()
            st.metric("Struggling Learners", struggling)
        else:
            st.metric("Struggling Learners", "N/A")

    with col3:
        if 'Predicted_HighRisk' in df.columns:
            high_risk = df['Predicted_HighRisk'].sum()
            st.metric("High Risk Students", high_risk)
        else:
            st.metric("High Risk Students", "N/A")

    with col4:
        if 'AttendanceRate' in df.columns:
            avg_attendance = df['AttendanceRate'].mean()
            st.metric("Average Attendance", f"{avg_attendance:.1%}")
        else:
            st.metric("Average Attendance", "N/A")

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'PerformanceCategory' in df.columns:
            performance_filter = st.selectbox(
                "Performance Category",
                ["All"] + list(df['PerformanceCategory'].unique())
            )
        else:
            performance_filter = "All"

    with col2:
        if 'Predicted_HighRisk' in df.columns:
            risk_filter = st.selectbox(
                "Risk Level",
                ["All", "Low Risk", "High Risk"]
            )
        else:
            risk_filter = "All"

    with col3:
        if 'Gender' in df.columns:
            gender_filter = st.selectbox(
                "Gender",
                ["All"] + list(df['Gender'].unique())
            )
        else:
            gender_filter = "All"

    # Apply filters
    filtered_df = df.copy()
    if performance_filter != "All" and 'PerformanceCategory' in df.columns:
        filtered_df = filtered_df[filtered_df['PerformanceCategory'] == performance_filter]
    if risk_filter != "All" and 'Predicted_HighRisk' in df.columns:
        risk_value = 1 if risk_filter == "High Risk" else 0
        filtered_df = filtered_df[filtered_df['Predicted_HighRisk'] == risk_value]
    if gender_filter != "All" and 'Gender' in df.columns:
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Performance Distribution
        if 'PerformanceCategory' in df.columns:
            perf_counts = filtered_df['PerformanceCategory'].value_counts()
            fig1 = px.pie(
                values=perf_counts.values,
                names=perf_counts.index,
                title="Performance Category Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Feature Importance
        if st.session_state.get('models_trained', False) and rf_perf is not None:
            feature_importance = pd.DataFrame({
                'feature': st.session_state.features,
                'importance': rf_perf.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)

            fig3 = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance (Random Forest)"
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Risk Distribution
        if 'Predicted_Risk_Probability' in df.columns:
            fig2 = px.histogram(
                filtered_df,
                x='Predicted_Risk_Probability',
                title="Risk Probability Distribution",
                nbins=20
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Academic Performance by Demographics
        if 'TestScore_Math' in df.columns and 'Gender' in df.columns:
            fig4 = px.box(
                filtered_df,
                x='Gender',
                y='TestScore_Math',
                title="Math Scores by Gender"
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Model Performance
    if st.session_state.get('models_trained', False):
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.perf_accuracy is not None:
                st.metric("Performance Classification Accuracy",
                          f"{st.session_state.perf_accuracy:.3f}")

        with col2:
            if st.session_state.risk_accuracy is not None:
                st.metric("Risk Prediction Accuracy",
                          f"{st.session_state.risk_accuracy:.3f}")

    # Student Details Table
    st.subheader("Student Details")
    display_cols = ['Age', 'Gender', 'TestScore_Math', 'TestScore_Reading',
                    'AttendanceRate', 'StudyHours']
    display_cols = [col for col in display_cols if col in filtered_df.columns]

    if 'PerformanceCategory' in filtered_df.columns:
        display_cols.append('PerformanceCategory')
    if 'Predicted_Risk_Probability' in filtered_df.columns:
        display_cols.append('Predicted_Risk_Probability')

    st.dataframe(filtered_df[display_cols].head(20), use_container_width=True)

    # Download processed data
    if st.button("Download Processed Data"):
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_student_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)


# Main app logic
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()


if __name__ == "__main__":
    main()