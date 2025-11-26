import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import io
import base64
import warnings
import os

warnings.filterwarnings('ignore')

# Clear all cache to force reload
st.cache_data.clear()
st.cache_resource.clear()

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
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffd93d;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #6bcf7f;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .filter-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .feature-impact {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# Hardcoded Login System
def check_login(username, password):
    valid_users = {"admin": "admin123",}
    return valid_users.get(username) == password

# Login Page
def login_page():
    st.markdown('<div class="main-header">🎓 Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")


# Load training and validation data (80% of total)
@st.cache_data(ttl=3600, show_spinner=False)
def load_training_validation_data():
    """Load the 80% training + validation data for model training"""
    try:
        # Check if data directory exists
        if not os.path.exists('data'):
            st.error("'data' folder not found! Please create it and add your CSV files.")
            return None, None, None

        # Load training data (64%)
        if not os.path.exists("data/studentdata_training.csv"):
            st.error("studentdata_training.csv not found!")
            return None, None, None

        df_train = pd.read_csv("data/studentdata_training.csv")
        st.success("✅ Loaded studentdata_training.csv (64%)")

        # Load validation data (16%)
        if not os.path.exists("data/studentdata_validation.csv"):
            st.error("studentdata_validation.csv not found!")
            return None, None, None

        df_val = pd.read_csv("data/studentdata_validation.csv")
        st.success("✅ Loaded studentdata_validation.csv (16%)")

        # SIMPLE CHECK - Only PerformanceCategory_Encoded
        if 'PerformanceCategory_Encoded' not in df_train.columns:
            st.error("❌ PerformanceCategory_Encoded column not found in training data!")
            st.info(f"Available columns: {list(df_train.columns)}")
            return None, None, None

        # Combine for training (80% total)
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
        st.success(f"✅ Combined training data: {len(df_combined):,} students (80%)")

        return df_combined, df_train, df_val

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


# Load Featured data for EDA - OPTIMIZED FOR 1M ROWS
@st.cache_data(ttl=3600, show_spinner=False)
def load_featured_data():
    """Load the featured data for EDA analysis with optimizations for large datasets"""
    try:
        if not os.path.exists("data/studentdata_featured.csv"):
            st.error("studentdata_featured.csv not found!")
            return None

        # Use optimized pandas reading for large datasets
        df_featured = pd.read_csv(
            "data/studentdata_featured.csv",
            dtype={
                'Gender': 'category',
                'ParentalEducation': 'category',
                'SchoolType': 'category',
                'PerformanceCategory': 'category'
            },
            usecols=lambda x: x != 'GPA'  # Skip GPA column during load
        )

        # Convert appropriate numeric columns to more efficient types
        int_cols = ['Age', 'SES_Quartile', 'Extracurricular', 'PartTimeJob', 'ParentSupport']
        for col in int_cols:
            if col in df_featured.columns:
                df_featured[col] = pd.to_numeric(df_featured[col], downcast='integer')

        float_cols = ['TestScore_Math', 'TestScore_Reading', 'TestScore_Science', 'AttendanceRate', 'StudyHours']
        for col in float_cols:
            if col in df_featured.columns:
                df_featured[col] = pd.to_numeric(df_featured[col], downcast='float')

        st.success(f"✅ Loaded full dataset: {len(df_featured):,} students")
        return df_featured

    except Exception as e:
        st.error(f"Error loading featured data: {str(e)}")
        return None


# Get consistent feature columns
def get_feature_columns(df):
    """Get consistent feature columns for training and prediction"""
    # SIMPLE: All columns except PerformanceCategory_Encoded
    feature_cols = [col for col in df.columns if col != 'PerformanceCategory_Encoded']
    return feature_cols


# Train Random Forest Model on 80% data
@st.cache_resource
def train_random_forest(df):
    try:
        # Get feature columns
        feature_cols = get_feature_columns(df)

        if not feature_cols:
            st.error("No suitable features found for training!")
            return None, None

        X = df[feature_cols]
        y = df['PerformanceCategory_Encoded']

        st.info(f"📊 Target distribution: {y.value_counts().to_dict()}")

        # Train model on full 80% data
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )

        rf_model.fit(X, y)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return rf_model, feature_importance

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None


# Validate model on validation set
def validate_model(model, df_val):
    """Validate model on the 16% validation set"""
    try:
        feature_cols = model.feature_names_in_
        X_val = df_val[feature_cols]
        y_val = df_val['PerformanceCategory_Encoded']

        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        return accuracy, cm, y_pred, y_pred_proba

    except Exception as e:
        st.error(f"Error validating model: {str(e)}")
        return None, None, None, None


# Make predictions on testing data
def predict_testing_data(_model, testing_df):
    """Make predictions on the uploaded testing data"""
    try:
        # Get the same features used during training
        feature_cols = _model.feature_names_in_

        # Check if all features are present
        missing_features = set(feature_cols) - set(testing_df.columns)
        if missing_features:
            st.error(f"Missing features in testing data: {missing_features}")
            return None

        # Select only the features used in training
        X_testing = testing_df[feature_cols]

        # Make predictions
        predictions = _model.predict(X_testing)
        prediction_probas = _model.predict_proba(X_testing)

        # Add predictions to dataframe
        result_df = testing_df.copy()
        result_df['Predicted_Performance'] = predictions
        result_df['Prediction_Confidence'] = np.max(prediction_probas, axis=1)

        # Map performance codes to labels
        performance_mapping = {1: 'High Achiever', 0: 'Average Performer', 2: 'Struggling Learner'}
        result_df['Predicted_Label'] = result_df['Predicted_Performance'].map(performance_mapping)
        result_df['Risk_Level'] = result_df['Predicted_Performance'].map({1: 'Low Risk', 0: 'Low Risk', 2: 'High Risk'})

        return result_df

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None


# Analyze feature impact for high-risk students
def analyze_high_risk_features(predictions_df, top_features):
    """Analyze feature values for high-risk students"""
    high_risk_df = predictions_df[predictions_df['Risk_Level'] == 'High Risk']
    low_risk_df = predictions_df[predictions_df['Risk_Level'] == 'Low Risk']

    feature_analysis = []

    for feature in top_features:
        if feature in predictions_df.columns:
            if predictions_df[feature].dtype in ['int64', 'float64']:
                # For numerical features
                high_risk_mean = high_risk_df[feature].mean()
                low_risk_mean = low_risk_df[feature].mean()
                feature_analysis.append({
                    'feature': feature,
                    'high_risk_mean': high_risk_mean,
                    'low_risk_mean': low_risk_mean,
                    'difference': high_risk_mean - low_risk_mean,
                    'type': 'numerical'
                })
            else:
                # For categorical features - get most common value
                high_risk_mode = high_risk_df[feature].mode()[0] if not high_risk_df[feature].mode().empty else 'N/A'
                low_risk_mode = low_risk_df[feature].mode()[0] if not low_risk_df[feature].mode().empty else 'N/A'
                feature_analysis.append({
                    'feature': feature,
                    'high_risk_mode': high_risk_mode,
                    'low_risk_mode': low_risk_mode,
                    'type': 'categorical'
                })

    return pd.DataFrame(feature_analysis)


# Create detailed feature analysis graphs
def create_feature_detail_graphs(predictions_df, top_features):
    """Create detailed individual graphs for top features"""
    graphs = {}

    for feature in top_features:
        if feature in predictions_df.columns:
            if predictions_df[feature].dtype in ['int64', 'float64']:
                # Create distribution plot by risk level
                fig = px.histogram(
                    predictions_df,
                    x=feature,
                    color='Risk_Level',
                    nbins=20,
                    title=f'{feature} Distribution by Risk Level',
                    color_discrete_map={'Low Risk': '#4ECDC4', 'High Risk': '#FF6B6B'},
                    opacity=0.7,
                    barmode='overlay'
                )
                fig.update_layout(
                    xaxis_title=feature,
                    yaxis_title='Number of Students',
                    legend_title='Risk Level'
                )
                graphs[feature] = fig

                # Create box plot
                fig_box = px.box(
                    predictions_df,
                    x='Risk_Level',
                    y=feature,
                    color='Risk_Level',
                    title=f'{feature} Distribution by Risk Level',
                    color_discrete_map={'Low Risk': '#4ECDC4', 'High Risk': '#FF6B6B'}
                )
                fig_box.update_layout(showlegend=False)
                graphs[f"{feature}_box"] = fig_box

            else:
                # For categorical features
                cross_tab = pd.crosstab(predictions_df[feature], predictions_df['Risk_Level'])
                fig = px.bar(
                    cross_tab,
                    x=cross_tab.index,
                    y=cross_tab.columns,
                    title=f'{feature} Distribution by Risk Level',
                    labels={'value': 'Count', 'variable': 'Risk Level'},
                    barmode='group',
                    color_discrete_sequence=['#4ECDC4', '#FF6B6B']
                )
                graphs[feature] = fig

    return graphs


# Main Dashboard
def main_dashboard():
    # Sidebar Navigation
    st.sidebar.title("🎓 Navigation")
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")

    # Sidebar tabs
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dashboard Sections")

    # Create sidebar navigation - ADDED EDA TAB
    tab_options = ["📈 Overview", "🔍 EDA Analysis", "🎯 Testing Predictions", "📋 Data Details"]
    selected_tab = st.sidebar.radio("Select Section", tab_options)

    # Data management in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Management")

    if st.sidebar.button("Reload Training Data", use_container_width=True):
        # Clear cache and reload
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'df_combined' in st.session_state:
            del st.session_state.df_combined
        if 'rf_model' in st.session_state:
            del st.session_state.rf_model
        if 'df_featured' in st.session_state:
            del st.session_state.df_featured
        st.rerun()

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    # Load training data (80%)
    if 'df_combined' not in st.session_state:
        with st.spinner("Loading training data..."):
            df_combined, df_train, df_val = load_training_validation_data()
            if df_combined is not None:
                st.session_state.df_combined = df_combined
                st.session_state.df_train = df_train
                st.session_state.df_val = df_val
            else:
                st.error("Failed to load training data. Please check your files.")
                return

    # Load Featured data for EDA - ONLY when needed
    if selected_tab == "🔍 EDA Analysis":
        if 'df_featured' not in st.session_state:
            with st.spinner("Loading featured data for EDA (this may take a moment for large datasets)..."):
                df_featured = load_featured_data()
                if df_featured is not None:
                    st.session_state.df_featured = df_featured

    # Train model if data is loaded
    if 'df_combined' in st.session_state and 'rf_model' not in st.session_state:
        with st.spinner("Training Random Forest model on 80% data..."):
            rf_model, feature_importance = train_random_forest(st.session_state.df_combined)
            if rf_model is not None:
                st.session_state.rf_model = rf_model
                st.session_state.feature_importance = feature_importance

                # Validate model
                accuracy, cm, y_pred, y_pred_proba = validate_model(rf_model, st.session_state.df_val)
                if accuracy is not None:
                    st.session_state.val_accuracy = accuracy
                    st.session_state.val_cm = cm
            else:
                st.error("Model training failed!")

    # Main dashboard content
    st.markdown('<div class="main-header">📊 Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)

    # Display selected tab content
    if selected_tab == "📈 Overview":
        display_overview_tab()
    elif selected_tab == "🔍 EDA Analysis":
        display_eda_tab()
    elif selected_tab == "🎯 Testing Predictions":
        display_testing_predictions_tab()
    elif selected_tab == "📋 Data Details":
        display_data_details_tab()


def display_overview_tab():
    st.markdown('<div class="section-header">📊 Performance Overview</div>', unsafe_allow_html=True)

    if 'df_combined' not in st.session_state:
        st.warning("Please ensure training data files are available in the 'data' folder.")
        return

    df_combined = st.session_state.df_combined

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        total_students = len(df_combined)
        st.metric("Training Students", f"{total_students:,}")

    with col2:
        struggling = (df_combined['PerformanceCategory_Encoded'] == 2).sum()
        st.metric("Struggling Learners", f"{struggling}")

    with col3:
        struggling_pct = (struggling / len(df_combined)) * 100
        st.metric("Struggling Percentage", f"{struggling_pct:.1f}%")

    # Performance Distribution
    col1, col2 = st.columns(2)

    with col1:
        performance_counts = df_combined['PerformanceCategory_Encoded'].value_counts().sort_index()
        performance_names = ['Average Performer', 'High Achiever', 'Struggling Learner']

        fig1 = px.pie(
            values=performance_counts.values,
            names=performance_names,
            title="🎓 Performance Category Distribution",
            color_discrete_sequence=['#4169E1', '#DC143C', '#2E8B57']
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        risk_counts = (df_combined['PerformanceCategory_Encoded'] == 2).value_counts()
        risk_labels = ['Not At Risk', 'At Risk']

        fig2 = px.pie(
            values=risk_counts.values,
            names=risk_labels,
            title="🚨 Risk Classification Distribution",
            color_discrete_sequence=['#32CD32', '#FF4444']
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ML Model Section directly in Overview - MOVED FROM ML MODEL TAB
    if 'rf_model' in st.session_state:

        rf_model = st.session_state.rf_model
        feature_importance = st.session_state.feature_importance

        # Validation Results
        if 'val_accuracy' in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                # Confusion Matrix
                fig_cm = px.imshow(
                    st.session_state.val_cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['High (1)', 'Average (0)', 'Struggling (2)'],
                    y=['High (1)', 'Average (0)', 'Struggling (2)'],
                    title="Confusion Matrix - Validation Set"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        st.markdown("### 🔍 Feature Importance Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            top_10_features = feature_importance.head(10)
            fig_fi = px.bar(
                top_10_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='importance',
                color_continuous_scale='Reds'
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)

        with col2:
            st.markdown("#### Feature Importance Table")
            st.dataframe(feature_importance.head(10), use_container_width=True)


def display_eda_tab():
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="color: black;">
    Analyzing all 1,000,000 records. Using optimized visualizations for large datasets.
    </div>
    """, unsafe_allow_html=True)

    # Load featured data
    if 'df_featured' not in st.session_state:
        st.warning("Please ensure featured data file is available at 'data/studentdata_featured.csv'")
        return

    df_featured = st.session_state.df_featured

    # Basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", f"{len(df_featured):,}")
    with col2:
        st.metric("Total Features", f"{len(df_featured.columns)}")
    with col3:
        performance_counts = df_featured['PerformanceCategory'].value_counts()
        st.metric("Performance Categories", len(performance_counts))

    # Performance Category Distribution - FAST with aggregated data
    st.markdown("### 🎯 Performance Category Distribution")

    col1, col2 = st.columns(2)

    with col1:
        performance_counts = df_featured['PerformanceCategory'].value_counts()
        fig_perf = px.pie(
            values=performance_counts.values,
            names=performance_counts.index,
            title="Performance Category Distribution",
            color_discrete_sequence=['#4169E1', '#DC143C', '#2E8B57']
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=performance_counts.index,
            y=performance_counts.values,
            title="Performance Category Counts",
            labels={'x': 'Performance Category', 'y': 'Number of Students'},
            color=performance_counts.index,  # Use category names for coloring
            color_discrete_sequence=['#4169E1', '#DC143C', '#2E8B57']  # Blue, Red, Green
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Get features to analyze (exclude PerformanceCategory and GPA)
    exclude_cols = ['PerformanceCategory', 'GPA']
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]

    # Separate numerical and categorical features
    numerical_cols = df_featured[feature_cols].select_dtypes(include=[np.number]).columns
    categorical_cols = df_featured[feature_cols].select_dtypes(include=['object', 'category']).columns

    # Define all numerical variables for unified dropdown
    continuous_vars = ['TestScore_Math', 'TestScore_Reading', 'TestScore_Science', 'AttendanceRate', 'StudyHours']
    discrete_numerical = [col for col in numerical_cols if col not in continuous_vars]

    # Combine all numerical features for unified dropdown
    all_numerical_features = continuous_vars + discrete_numerical

    # Unified Numerical Features Analysis - USING BAR CHARTS
    if len(all_numerical_features) > 0:
        st.markdown("#### 📈 Numerical Features Analysis")

        selected_numerical = st.selectbox("Select numerical feature to analyze:", all_numerical_features)

        if selected_numerical:
            # Handle continuous variables (Test scores, Attendance, Study hours)
            if selected_numerical in ['TestScore_Math', 'TestScore_Reading', 'TestScore_Science']:
                # Test scores: create bins from 0-100 in steps of 10
                bins = list(range(0, 101, 10))
                labels = [f"{i}-{i + 9}" for i in range(0, 100, 10)]
                x_label = "Score Range"
                title_suffix = "Test Score Ranges"

            elif selected_numerical == 'AttendanceRate':
                # Attendance rate: create bins from 0.70-1.00 in steps of 0.05
                bins = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
                labels = ["0.70-0.74", "0.75-0.79", "0.80-0.84", "0.85-0.89", "0.90-0.94", "0.95-1.00"]
                x_label = "Attendance Rate Range"
                title_suffix = "Attendance Rate Ranges"

            elif selected_numerical == 'StudyHours':
                # Study hours: create bins from 0.0-4.00 in steps of 0.5
                bins = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
                labels = ["0.0-0.4", "0.5-0.9", "1.0-1.4", "1.5-1.9", "2.0-2.4", "2.5-2.9", "3.0-3.4", "3.5-4.0"]
                x_label = "Study Hours Range"
                title_suffix = "Study Hours Ranges"

            # Handle discrete numerical variables
            elif selected_numerical == 'SES_Quartile':
                # SES_Quartile is already 1,2,3,4 - use as is
                df_featured['binned'] = df_featured[selected_numerical].astype(str)
                x_labels = ['1', '2', '3', '4']
                title_suffix = "SES Quartile"
                plot_data = df_featured

            elif selected_numerical == 'Age':
                # Bin age into 14,15,16,17,18
                df_featured['binned'] = df_featured[selected_numerical].astype(str)
                # Filter to only show ages 14-18
                plot_data = df_featured[df_featured[selected_numerical].between(14, 18)]
                x_labels = ['14', '15', '16', '17', '18']
                title_suffix = "Age"

            elif selected_numerical in ['Extracurricular', 'PartTimeJob', 'ParentSupport']:
                # These are binary (0,1) - use as is
                df_featured['binned'] = df_featured[selected_numerical].astype(str)
                x_labels = ['0', '1']
                title_suffix = selected_numerical
                plot_data = df_featured

            else:
                # For other numerical features, use quartiles
                df_featured['binned'] = pd.cut(df_featured[selected_numerical], bins=5, precision=0).astype(str)
                x_labels = sorted(df_featured['binned'].unique())
                title_suffix = f"{selected_numerical} (Binned)"
                plot_data = df_featured

            # For continuous variables, create binned data
            if selected_numerical in continuous_vars:
                df_featured['binned'] = pd.cut(df_featured[selected_numerical], bins=bins, labels=labels,
                                               include_lowest=True)
                plot_data = df_featured

            # Calculate frequency distribution by performance category
            bar_data = plot_data.groupby(['binned', 'PerformanceCategory']).size().reset_index(name='Frequency')

            # Display only the count graph (removed percentage graph)
            fig_bar_counts = px.bar(
                bar_data,
                x='binned',
                y='Frequency',
                color='PerformanceCategory',
                title=f"{selected_numerical} Distribution by Performance Category",
                labels={'binned': x_label if selected_numerical in continuous_vars else selected_numerical,
                        'Frequency': 'Number of Students'},
                barmode='stack',
                color_discrete_sequence=['#4169E1', '#2E8B57', '#DC143C'],  # Green, Blue, Red
                category_orders={"binned": labels if selected_numerical in continuous_vars else x_labels}
            )
            if selected_numerical in continuous_vars:
                fig_bar_counts.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': labels},
                    hovermode='x unified'
                )
            st.plotly_chart(fig_bar_counts, use_container_width=True)

    # Categorical Features vs Performance Category - FAST with crosstabs
    if len(categorical_cols) > 0:
        st.markdown("#### 📊 Categorical Features Analysis")

        selected_cat = st.selectbox("Select categorical feature to analyze:", categorical_cols)

        if selected_cat:
            # Use crosstab for fast aggregation
            cross_tab = pd.crosstab(df_featured[selected_cat], df_featured['PerformanceCategory'])

            # Show top 15 categories to avoid overcrowding
            top_categories = cross_tab.sum(axis=1).nlargest(15)
            cross_tab_top = cross_tab.loc[top_categories.index]

            # Display only the count graph (removed percentage graph)
            fig_stacked = px.bar(
                cross_tab_top,
                x=cross_tab_top.index,
                y=cross_tab_top.columns,
                title=f"Top 15 {selected_cat} vs Performance Category",
                labels={'value': 'Count', 'variable': 'Performance Category'},
                barmode='stack',
                color_discrete_sequence=['#4169E1', '#2E8B57', '#DC143C']
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

    # Fast correlation analysis using sampling
    if len(numerical_cols) > 1:
        st.markdown("### 🔗 Correlation Analysis")

        # Use sample for correlation to make it fast
        corr_sample_size = min(50000, len(df_featured))
        df_corr_sample = df_featured.sample(n=corr_sample_size, random_state=42)

        corr_matrix = df_corr_sample[numerical_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Data preview with pagination
    st.markdown("### 📋 Data Preview")
    page_size = st.selectbox("Rows per page:", [100, 500, 1000], index=0)
    page_number = st.number_input("Page number:", min_value=1, value=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(df_featured.iloc[start_idx:end_idx], use_container_width=True)
    st.write(f"Showing rows {start_idx + 1} to {min(end_idx, len(df_featured))} of {len(df_featured):,}")


def display_testing_predictions_tab():
    st.markdown('<div class="section-header">🎯 Testing Predictions</div>', unsafe_allow_html=True)

    if 'rf_model' not in st.session_state:
        st.warning("Please wait for the model to finish training in the Overview tab.")
        return

    st.markdown("""
    <div class="info-box" style="color: black;">
    Upload your 20% testing data here to get predictions. 
    This should be the holdout set that was not used for training the model.
    File should contain the same features as training data.
    </div>
    """, unsafe_allow_html=True)

    # File upload for testing data
    uploaded_testing = st.file_uploader(
        "Choose testing data CSV file",
        type="csv",
        key="testing_upload"
    )

    if uploaded_testing is not None:
        try:
            testing_df = pd.read_csv(uploaded_testing)

            # Check if target column exists (optional)
            has_target = 'PerformanceCategory_Encoded' in testing_df.columns

            st.session_state.testing_data = testing_df
            st.success(f"✅ Testing data uploaded: {len(testing_df)} students")

        except Exception as e:
            st.error(f"Error reading testing file: {str(e)}")

    # Make predictions if testing data is uploaded
    if 'testing_data' in st.session_state:

        if st.button("Predict Performance on Testing Data", use_container_width=True, type="primary"):
            with st.spinner("Making predictions..."):
                predictions_df = predict_testing_data(st.session_state.rf_model, st.session_state.testing_data)

                if predictions_df is not None:
                    st.session_state.predictions = predictions_df
                    st.success("✅ Predictions completed!")

    # 🔥 KEY FIX: Always show existing predictions when available
    if 'predictions' in st.session_state:
        display_prediction_results(st.session_state.predictions)


def display_prediction_results(predictions_df):
    st.markdown('<div class="section-header">Comprehensive Prediction Analysis</div>', unsafe_allow_html=True)

    # 1. Overall Prediction Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        total_students = len(predictions_df)
        st.metric("Total Students Predicted", f"{total_students:,}")

    with col2:
        high_risk = (predictions_df['Predicted_Performance'] == 2).sum()
        st.metric("High Risk Predictions", f"{high_risk}")

    with col3:
        high_risk_pct = (high_risk / len(predictions_df)) * 100
        st.metric("High Risk Percentage", f"{high_risk_pct:.1f}%")

    # 2. Performance Distribution Overview
    st.markdown("### 📈 Performance Distribution Overview")
    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution with donut chart
        risk_counts = predictions_df['Risk_Level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Predicted Risk Level Distribution",
            color_discrete_sequence=['#32CD32', '#FF4444'],
            hole=0.4
        )
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        # Performance level distribution
        performance_counts = predictions_df['Predicted_Label'].value_counts()
        fig_perf = px.bar(
            x=performance_counts.index,
            y=performance_counts.values,
            title="Performance Levels",
            color=performance_counts.index,
            color_discrete_map={
                'High Achiever': '#2E8B57',
                'Average Performer': '#4169E1',
                'Struggling Learner': '#DC143C'
            },
            labels={'x': 'Performance Level', 'y': 'Number of Students'}
        )
        fig_perf.update_layout(showlegend=False)
        st.plotly_chart(fig_perf, use_container_width=True)

    # 3. Top 5 Most Influential Features for Risk Prediction
    st.markdown("### 🔥 Top 5 Most Influential Features for Risk Prediction")

    if 'feature_importance' in st.session_state:
        top_5_features = st.session_state.feature_importance.head(5)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Visualize top 5 features
            fig_top_features = px.bar(
                top_5_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 5 Features Driving Risk Predictions',
                color='importance',
                color_continuous_scale='Reds',
                labels={'importance': 'Feature Importance', 'feature': ''}
            )
            fig_top_features.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400
            )
            st.plotly_chart(fig_top_features, use_container_width=True)

        with col2:
            st.markdown("Feature Impact Summary")
            st.dataframe(
                top_5_features[['feature', 'importance']].round(4),
                use_container_width=True,
                hide_index=True
            )

            # Add insights about top features
            st.markdown("""
            **Key Insights:**
            - These features have the strongest impact on risk classification
            - Focus interventions on students with concerning values in these areas
            - Monitor these metrics proactively
            """)
    else:
        st.warning("Feature importance data not available.")

    # 4. Detailed Feature Analysis
    st.markdown("### 🔍 Detailed Feature Analysis")

    if 'feature_importance' in st.session_state and 'predictions' in st.session_state:
        top_features = st.session_state.feature_importance.head(5)['feature'].tolist()

        # Create detailed graphs for each top feature
        feature_graphs = create_feature_detail_graphs(predictions_df, top_features)

        # Create tabs for each feature
        feature_tabs = st.tabs([f"📈 {feature}" for feature in top_features])

        for i, (feature, tab) in enumerate(zip(top_features, feature_tabs)):
            with tab:
                col1, col2 = st.columns(2)

                with col1:
                    if feature in feature_graphs:
                        st.plotly_chart(feature_graphs[feature], use_container_width=True)

                with col2:
                    if f"{feature}_box" in feature_graphs:
                        st.plotly_chart(feature_graphs[f"{feature}_box"], use_container_width=True)

                # Move insights to the middle (full width below both charts)
                st.markdown(f"{feature} Insights")
                if feature in ['AttendanceRate', 'StudyHours', 'TestScore_Math', 'TestScore_Reading',
                               'TestScore_Science']:
                    if feature == 'AttendanceRate':
                        st.markdown("""
                        **Key Observations:**
                        - Lower attendance strongly correlates with higher risk
                        - Students with <80% attendance need immediate attention
                        - Regular attendance monitoring is crucial
                        """)
                    elif feature == 'StudyHours':
                        st.markdown("""
                        **Key Observations:**
                        - Insufficient study time increases risk significantly
                        - Students studying <1.5 hours daily are at higher risk
                        - Study habits intervention needed
                        """)
                    elif 'TestScore' in feature:
                        st.markdown("""
                        **Key Observations:**
                        - Test scores are strong predictors of academic risk
                        - Scores below 60 indicate high intervention need
                        - Subject-specific support required
                        """)
                else:
                    st.markdown("""
                    **Key Observations:**
                    - This feature significantly impacts risk prediction
                    - Monitor values outside normal ranges
                    - Consider targeted interventions
                    """)

        # Feature Values Analysis for High-Risk Students
        high_risk_df = predictions_df[predictions_df['Risk_Level'] == 'High Risk']

        if len(high_risk_df) > 0 and 'feature_importance' in st.session_state:
            top_features = st.session_state.feature_importance.head(5)['feature'].tolist()
            feature_analysis = analyze_high_risk_features(predictions_df, top_features)

            # Create comparison table
            comparison_data = []
            for _, row in feature_analysis.iterrows():
                if row['type'] == 'numerical':
                    comparison_data.append({
                        'Feature': row['feature'],
                        'High Risk Avg': f"{row['high_risk_mean']:.2f}",
                        'Low Risk Avg': f"{row['low_risk_mean']:.2f}",
                        'Difference': f"{row['difference']:.2f}",
                        'Impact': 'Higher' if row['difference'] > 0 else 'Lower'
                    })
                else:
                    comparison_data.append({
                        'Feature': row['feature'],
                        'High Risk Mode': row['high_risk_mode'],
                        'Low Risk Mode': row['low_risk_mode'],
                        'Difference': 'Categorical',
                        'Impact': 'Different Pattern'
                    })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

    # 5. Prediction Confidence Analysis
    st.markdown("### 🔍 Prediction Confidence Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Confidence distribution by risk level
        fig_confidence_hist = px.histogram(
            predictions_df,
            x='Prediction_Confidence',
            color='Risk_Level',
            title='Confidence Distribution by Risk Level',
            nbins=20,
            color_discrete_map={'Low Risk': '#32CD32', 'High Risk': '#FF4444'},
            opacity=0.7
        )
        fig_confidence_hist.update_layout(barmode='overlay')
        st.plotly_chart(fig_confidence_hist, use_container_width=True)

    with col2:
        # Confidence vs Performance Level
        fig_confidence_box = px.box(
            predictions_df,
            x='Predicted_Label',
            y='Prediction_Confidence',
            color='Predicted_Label',
            title='Confidence Distribution by Performance Level',
            color_discrete_map={
                'High Achiever': '#2E8B57',
                'Average Performer': '#4169E1',
                'Struggling Learner': '#DC143C'
            }
        )
        fig_confidence_box.update_layout(showlegend=False)
        st.plotly_chart(fig_confidence_box, use_container_width=True)

    # 6. High-Risk Students Analysis
    st.markdown("### 🚨 Confidence Distribution in High-Risk Predictions")

    high_risk_df = predictions_df[predictions_df['Risk_Level'] == 'High Risk']

    if len(high_risk_df) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total High-Risk Students", len(high_risk_df))

        with col2:
            avg_high_risk_confidence = high_risk_df['Prediction_Confidence'].mean()
            st.metric("Avg Confidence (High Risk)", f"{avg_high_risk_confidence:.3f}")

        with col3:
            low_confidence_high_risk = len(high_risk_df[high_risk_df['Prediction_Confidence'] < 0.7])
            st.metric("Low Confidence Predictions", low_confidence_high_risk)

        # High-risk students by confidence levels
        confidence_bins = pd.cut(high_risk_df['Prediction_Confidence'],
                                 bins=[0, 0.6, 0.8, 1.0],
                                 labels=['Low (0-0.6)', 'Medium (0.6-0.8)', 'High (0.8-1.0)'])
        confidence_counts = confidence_bins.value_counts()

        fig_confidence_high_risk = px.pie(
            values=confidence_counts.values,
            names=confidence_counts.index,
            color_discrete_sequence=['#FF6B6B', '#FFA726', '#66BB6A']
        )
        st.plotly_chart(fig_confidence_high_risk, use_container_width=True)

    else:
        st.success("🎉 No high-risk students identified in this dataset!")

    if len(high_risk_df) > 0:
        st.warning(
            f"🚨 **Immediate Attention Required:** {len(high_risk_df)} students identified as high-risk need intervention strategies.")


def display_data_details_tab():
    st.markdown('<div class="section-header">📋 Student Details</div>', unsafe_allow_html=True)

    # Check if testing data is available, otherwise use training data
    if 'testing_data' in st.session_state:
        df = st.session_state.testing_data
        data_source = "Testing Data"
        st.info(f"Showing Testing Data: {len(df):,} records (20% holdout)")
    elif 'df_combined' in st.session_state:
        df = st.session_state.df_combined
        data_source = "Training Data"
        st.info(f"Showing Training Data: {len(df):,} records (80% training)")
    else:
        st.warning("No data loaded yet.")
        return

    # Filters
    st.markdown("### Filters")
    col1, col2 = st.columns(2)

    with col1:
        performance_filter = st.selectbox(
            "Performance Category",
            ["All", "Average Performer", "High Achiever", "Struggling Learner"]
        )

    with col2:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "Low Risk", "High Risk"]
        )

    # Apply filters
    filtered_df = df.copy()

    # Check if PerformanceCategory_Encoded exists in the data
    if 'PerformanceCategory_Encoded' in df.columns:
        if performance_filter != "All":
            performance_mapping_reverse = {'Average Performer': 0, 'High Achiever': 1, 'Struggling Learner': 2}
            filtered_df = filtered_df[
                filtered_df['PerformanceCategory_Encoded'] == performance_mapping_reverse[performance_filter]]

        if risk_filter != "All":
            if risk_filter == "High Risk":
                filtered_df = filtered_df[filtered_df['PerformanceCategory_Encoded'] == 2]
            else:  # Low Risk
                filtered_df = filtered_df[filtered_df['PerformanceCategory_Encoded'].isin([0, 1])]
    else:
        st.info("ℹ️ Performance filtering not available for this dataset")

    # Display student data
    st.markdown(f"### {data_source} - Filtered Records ({len(filtered_df):,} students)")

    # Select columns to display
    display_cols = [col for col in df.columns]
    # Remove PerformanceCategory_Encoded if it exists
    if 'PerformanceCategory_Encoded' in display_cols:
        display_cols.remove('PerformanceCategory_Encoded')

    default_cols = display_cols[:8]  # Show first 8 columns by default

    selected_cols = st.multiselect(
        "Select columns to display:",
        display_cols,
        default=default_cols
    )

    if selected_cols:
        st.dataframe(filtered_df[selected_cols], use_container_width=True)

        # Simple download button
        if st.button("📥 Download Filtered Data as CSV", type="primary", use_container_width=True):
            csv = filtered_df[selected_cols].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            filename = f"student_{data_source.lower().replace(' ', '_')}.csv"
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Click here to download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success(f"✅ Download ready! {len(filtered_df):,} records from {data_source}")
    else:
        st.warning("Please select at least one column to display.")


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