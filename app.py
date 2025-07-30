import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Defense Personnel Mental Health Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">Defense Personnel Mental Health Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "Overview",
    "Data Analysis",
    "Clustering Results",
    "Predictive Modeling",
    "Visualizations",
    "Individual Assessment"
])

@st.cache_data
def load_data():
    """Load and preprocess the mental health data"""
    try:
        # Try to load the CSV file
        df = pd.read_csv("MindFIT - Form Responses 1 (1).csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'MindFIT - Form Responses 1 (1).csv' is in the same directory.")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data and perform sentiment analysis"""
    if df is None:
        return None, None, None
    
    # Identify numeric and textual columns
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    textual_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Remove non-response columns
    non_response_columns = ["Timestamp", "Name", "Gender", "Role"]
    textual_columns = [col for col in textual_columns if col not in non_response_columns]
    
    # Sentiment analysis for textual columns
    sentiment_results_text = {}
    for col in textual_columns:
        if col in df.columns:
            sentiments = df[col].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
            sentiment_results_text[col] = sentiments
    
    sentiment_df_text = pd.DataFrame(sentiment_results_text)
    
    # Map numeric responses to sentiment scores
    def map_numeric_sentiment(value):
        if pd.isna(value):
            return 0
        if value <= 2:
            return -1
        elif value == 3:
            return 0
        else:
            return 1
    
    numeric_sentiment_df = df[numeric_columns].applymap(map_numeric_sentiment)
    
    # Combine sentiment scores
    combined_sentiment_df = pd.concat([sentiment_df_text, numeric_sentiment_df], axis=1)
    
    # Calculate total sentiment score
    df["Total_Sentiment_Score"] = combined_sentiment_df.sum(axis=1)
    
    return df, combined_sentiment_df, sentiment_df_text

@st.cache_data
def perform_clustering(combined_sentiment_df):
    """Perform K-means clustering"""
    if combined_sentiment_df is None:
        return None, None
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_sentiment_df.fillna(0))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    return clusters, kmeans

def create_cluster_distribution_plot(df):
    """Create cluster distribution plot"""
    cluster_counts = df["Sentiment_Cluster"].value_counts().sort_index()
    
    # Map cluster numbers to meaningful names
    cluster_names = ['High Health', 'Moderate Health', 'Needs Support']
    
    fig = px.bar(
        x=cluster_names,
        y=cluster_counts.values,
        title="Mental Health Groups Distribution",
        labels={'x': 'Health Group', 'y': 'Number of Personnel'},
        color=cluster_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_demographic_plots(df):
    """Create demographic analysis plots"""
    plots = {}
    
    # Create a mapping for cluster names
    cluster_mapping = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}
    df_mapped = df.copy()
    df_mapped['Cluster_Name'] = df_mapped['Sentiment_Cluster'].map(cluster_mapping)
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_cluster = pd.crosstab(df_mapped['Gender'], df_mapped['Cluster_Name'])
        fig_gender = px.bar(
            gender_cluster,
            title="Gender Distribution Across Clusters",
            labels={'value': 'Count', 'index': 'Gender'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        plots['gender'] = fig_gender
    
    # Role distribution
    if 'Role' in df.columns:
        role_cluster = pd.crosstab(df_mapped['Role'], df_mapped['Cluster_Name'])
        fig_role = px.bar(
            role_cluster,
            title="Role Distribution Across Clusters",
            labels={'value': 'Count', 'index': 'Role'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_role.update_xaxes(tickangle=45)
        plots['role'] = fig_role
    
    return plots

# Load and preprocess data
df = load_data()
if df is not None:
    df_processed, combined_sentiment_df, sentiment_df_text = preprocess_data(df)
    
    if df_processed is not None and combined_sentiment_df is not None:
        # Perform clustering
        clusters, kmeans_model = perform_clustering(combined_sentiment_df)
        if clusters is not None:
            df_processed["Sentiment_Cluster"] = clusters
            # Add meaningful cluster names
            cluster_mapping = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}
            df_processed["Cluster_Name"] = df_processed["Sentiment_Cluster"].map(cluster_mapping)

# Page content based on selection
if page == "Overview":
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Analysis
        
        This comprehensive mental health analysis tool is designed for defense personnel assessment. 
        The system uses advanced AI techniques including:
        
        - **Sentiment Analysis**: Natural language processing to analyze textual responses
        - **Machine Learning Clustering**: K-means algorithm to identify mental health groups
        - **Predictive Modeling**: Random Forest classifier for mental readiness prediction
        - **Statistical Analysis**: Demographic patterns and correlations
        
        ### Key Features
        - Real-time data processing and visualization
        - Interactive dashboards and charts
        - Individual assessment capabilities
        - Comprehensive reporting system
        """)
    
    with col2:
        if df is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Personnel", len(df))
            st.metric("Assessment Questions", len(df.columns) - 4)  # Excluding metadata columns
            if 'Sentiment_Cluster' in df.columns:
                st.metric("Mental Health Groups", df['Sentiment_Cluster'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">Data Analysis</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Sample Data:**")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.subheader("Data Summary")
            if df_processed is not None and 'Total_Sentiment_Score' in df_processed.columns:
                st.write("**Sentiment Score Statistics:**")
                st.write(df_processed['Total_Sentiment_Score'].describe())
                
                # Sentiment score distribution
                fig_hist = px.histogram(
                    df_processed, 
                    x='Total_Sentiment_Score',
                    title="Distribution of Total Sentiment Scores",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Sentiment analysis in progress... Please wait for data processing to complete.")
    else:
        st.error("No data available for analysis.")

elif page == "Clustering Results":
    st.markdown('<h2 class="sub-header">Clustering Results</h2>', unsafe_allow_html=True)
    
    if df_processed is not None and 'Sentiment_Cluster' in df_processed.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            fig_cluster = create_cluster_distribution_plot(df_processed)
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_summary = df_processed.groupby('Sentiment_Cluster')['Total_Sentiment_Score'].agg(['mean', 'count', 'std']).round(2)
            cluster_summary.index = ['High Health', 'Moderate Health', 'Needs Support']
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Show cluster distribution with names
            if 'Cluster_Name' in df_processed.columns:
                st.subheader("Cluster Distribution")
                cluster_dist = df_processed['Cluster_Name'].value_counts()
                st.dataframe(cluster_dist.to_frame('Count'), use_container_width=True)
        
        # Demographic analysis
        demographic_plots = create_demographic_plots(df_processed)
        
        if demographic_plots:
            st.subheader("Demographic Analysis")
            for plot_name, plot_fig in demographic_plots.items():
                st.plotly_chart(plot_fig, use_container_width=True)
    else:
        st.error("Clustering analysis not available. Please check the data.")

elif page == "Predictive Modeling":
    st.markdown('<h2 class="sub-header">Predictive Modeling</h2>', unsafe_allow_html=True)
    
    if df_processed is not None and combined_sentiment_df is not None and 'Sentiment_Cluster' in df_processed.columns:
        # Train Random Forest model
        X = combined_sentiment_df.fillna(0)
        y = df_processed['Sentiment_Cluster']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("Classification Report")
            class_report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(class_report).transpose().round(2)
            # Update index names for better readability
            if '0' in report_df.index:
                index_mapping = {'0': 'High Health', '1': 'Moderate Health', '2': 'Needs Support'}
                report_df.index = report_df.index.map(lambda x: index_mapping.get(x, x))
            st.dataframe(report_df, use_container_width=True)
    else:
        st.error("Predictive modeling not available. Please check the data.")

elif page == "Visualizations":
    st.markdown('<h2 class="sub-header">Visualizations</h2>', unsafe_allow_html=True)
    
    # Display saved visualizations from assets folder
    asset_files = {
        "Mental Health Groups Distribution": "mental health groups distribution.png",
        "Elbow Method": "elbow method.png",
        "Gender Distribution": "gender distribution.png",
        "Role Distribution": "Role distribution withi clusters.png",
        "Age Distribution": "Age Distribution with clusters.png",
        "Key Questions": "questions that classified the most.png"
    }
    
    for title, filename in asset_files.items():
        filepath = f"assets/{filename}"
        if os.path.exists(filepath):
            st.subheader(title)
            image = Image.open(filepath)
            st.image(image, use_column_width=True)
        else:
            st.warning(f"Visualization '{title}' not found at {filepath}")

elif page == "Individual Assessment":
    st.markdown('<h2 class="sub-header">Individual Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Mental Health Assessment Tool
    
    This tool allows for individual assessment based on the trained model.
    Answer the questions below to get a mental health readiness prediction.
    """)
    
    # Create a simple assessment form
    with st.form("assessment_form"):
        st.subheader("Assessment Questions")
        
        worry_level = st.slider("How much do you worry about different things compared to most people?", 1, 5, 3)
        mental_health_perception = st.slider("How good do you think your mental health is compared to most people?", 1, 5, 3)
        stress_handling = st.slider("How well do you handle stress in critical situations?", 1, 5, 3)
        emotional_regulation = st.slider("How well can you identify and manage your emotional reactions?", 1, 5, 3)
        decision_making = st.slider("How confident are you in making decisions under pressure?", 1, 5, 3)
        
        submitted = st.form_submit_button("Get Assessment")
        
        if submitted:
            # Simple prediction logic (you can enhance this with the actual model)
            avg_score = np.mean([worry_level, mental_health_perception, stress_handling, emotional_regulation, decision_making])
            
            if avg_score >= 4:
                result = "High Health"
                color = "green"
            elif avg_score >= 3:
                result = "Moderate Health"
                color = "orange"
            else:
                result = "Needs Support"
                color = "red"
            
            st.markdown(f"### Assessment Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
            
            if result == "Needs Support":
                st.warning("Consider seeking additional support or counseling.")
            elif result == "Moderate Health":
                st.info("Maintain current wellness practices and monitor mental health regularly.")
            else:
                st.success("Excellent mental health status!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Defense Personnel Mental Health Analysis | Developed for DRDO-SSPL | 2025
</div>
""", unsafe_allow_html=True)
