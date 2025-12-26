import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    h1 { color: #1f77b4; margin-bottom: 0; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDownloadButton button {
        background: #10b981;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Predictive Analytics Dashboard")
st.markdown("**Build, Train, and Deploy ML Models in Real-Time**")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    
    data_source = st.radio(
        "Data Source",
        ["Sample Dataset", "Upload CSV"],
        help="Choose to use built-in sample data or upload your own"
    )
    
    st.divider()
    
    if data_source == "Sample Dataset":
        dataset_choice = st.selectbox(
            "Select Sample Dataset",
            ["Sales Forecast", "Housing Prices", "Customer Churn"]
        )
    
    st.divider()
    st.markdown("### 🎯 Model Settings")
    model_type = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Linear Regression"]
    )
    
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5)
    
    if model_type in ["Random Forest", "Gradient Boosting"]:
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 3, 20, 10, 1)

# Function to generate sample data
@st.cache_data
def generate_sample_data(dataset_type):
    np.random.seed(42)
    n_samples = 1000
    
    if dataset_type == "Sales Forecast":
        data = pd.DataFrame({
            'Marketing_Spend': np.random.uniform(1000, 50000, n_samples),
            'Social_Media_Engagement': np.random.uniform(100, 10000, n_samples),
            'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
            'Competitor_Price': np.random.uniform(20, 100, n_samples),
            'Website_Traffic': np.random.uniform(500, 20000, n_samples)
        })
        # Create target with realistic relationships
        data['Sales'] = (
            0.5 * data['Marketing_Spend'] +
            0.3 * data['Social_Media_Engagement'] +
            0.1 * data['Website_Traffic'] -
            100 * data['Competitor_Price'] +
            np.random.normal(0, 5000, n_samples)
        )
        target = 'Sales'
        
    elif dataset_type == "Housing Prices":
        data = pd.DataFrame({
            'Square_Feet': np.random.uniform(800, 4000, n_samples),
            'Bedrooms': np.random.randint(1, 6, n_samples),
            'Bathrooms': np.random.randint(1, 4, n_samples),
            'Age_Years': np.random.randint(0, 50, n_samples),
            'Distance_to_City': np.random.uniform(1, 30, n_samples)
        })
        data['Price'] = (
            200 * data['Square_Feet'] +
            20000 * data['Bedrooms'] +
            15000 * data['Bathrooms'] -
            500 * data['Age_Years'] -
            2000 * data['Distance_to_City'] +
            np.random.normal(0, 30000, n_samples)
        )
        target = 'Price'
        
    else:  # Customer Churn
        data = pd.DataFrame({
            'Monthly_Charges': np.random.uniform(20, 150, n_samples),
            'Tenure_Months': np.random.randint(1, 72, n_samples),
            'Total_Charges': np.random.uniform(100, 8000, n_samples),
            'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
            'Support_Tickets': np.random.randint(0, 10, n_samples)
        })
        data['Churn_Score'] = (
            2 * data['Monthly_Charges'] -
            0.5 * data['Tenure_Months'] +
            5 * data['Support_Tickets'] +
            np.random.normal(50, 20, n_samples)
        )
        target = 'Churn_Score'
    
    return data, target

# Load data
if data_source == "Sample Dataset":
    df, target_col = generate_sample_data(dataset_choice)
    st.success(f"✅ Loaded {dataset_choice} dataset with {len(df)} records")
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded dataset with {len(df)} records")
        target_col = st.selectbox("Select Target Variable", df.columns)
    else:
        st.info("👆 Please upload a CSV file to continue")
        st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Explorer", "🤖 Model Training", "🎯 Predictions", "📊 Reports"])

with tab1:
    st.subheader("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Target Variable", target_col)
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📊 Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True, height=400)
    
    st.divider()
    
    # Visualizations
    st.markdown("### 📉 Feature Distributions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox("Select Feature for Distribution", numeric_cols)
        fig = px.histogram(df, x=selected_feature, nbins=30, 
                          title=f"Distribution of {selected_feature}",
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(numeric_cols) > 1:
            x_feature = st.selectbox("X-axis", numeric_cols, index=0)
            y_feature = st.selectbox("Y-axis", [target_col])
            
            fig = px.scatter(df, x=x_feature, y=y_feature, 
                           title=f"{x_feature} vs {y_feature}",
                           trendline="ols",
                           color_discrete_sequence=['#764ba2'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🤖 Model Training & Evaluation")
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size/100, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            # Select model
            if model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "Gradient Boosting":
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                model = LinearRegression()
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = X_encoded.columns.tolist()
            st.session_state['metrics'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
            st.session_state['predictions'] = {
                'y_test': y_test,
                'y_pred': y_pred_test
            }
            
        st.success("✅ Model trained successfully!")
    
    # Display results if model exists
    if 'model' in st.session_state:
        st.divider()
        
        metrics = st.session_state['metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>R² Score</h3>
                <h1>{:.3f}</h1>
                <p>Test Set</p>
            </div>
            """.format(metrics['test_r2']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>RMSE</h3>
                <h1>{:,.0f}</h1>
                <p>Test Set</p>
            </div>
            """.format(metrics['test_rmse']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>MAE</h3>
                <h1>{:,.0f}</h1>
                <p>Test Set</p>
            </div>
            """.format(metrics['test_mae']), unsafe_allow_html=True)
        
        st.divider()
        
        # Detailed metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Training Metrics")
            train_metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Value': [
                    f"{metrics['train_r2']:.4f}",
                    f"{metrics['train_rmse']:,.2f}",
                    f"{metrics['train_mae']:,.2f}"
                ]
            })
            st.dataframe(train_metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📊 Test Metrics")
            test_metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Value': [
                    f"{metrics['test_r2']:.4f}",
                    f"{metrics['test_rmse']:,.2f}",
                    f"{metrics['test_mae']:,.2f}"
                ]
            })
            st.dataframe(test_metrics_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Prediction visualization
        st.markdown("### 🎯 Actual vs Predicted")
        
        pred_data = st.session_state['predictions']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_data['y_test'],
            y=pred_data['y_pred'],
            mode='markers',
            name='Predictions',
            marker=dict(color='#667eea', size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(pred_data['y_test'].min(), pred_data['y_pred'].min())
        max_val = max(pred_data['y_test'].max(), pred_data['y_pred'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if applicable)
        if model_type in ["Random Forest", "Gradient Boosting"]:
            st.divider()
            st.markdown("### 🔍 Feature Importance")
            
            model = st.session_state['model']
            feature_names = st.session_state['feature_names']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h',
                        title="Top 10 Most Important Features",
                        color='Importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🎯 Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")
    else:
        st.markdown("### Enter Feature Values")
        
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        
        # Create input fields
        input_data = {}
        
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            with cols[idx % 3]:
                # Get reasonable default from training data
                if feature in X_encoded.columns:
                    default_val = float(X_encoded[feature].median())
                    min_val = float(X_encoded[feature].min())
                    max_val = float(X_encoded[feature].max())
                    
                    input_data[feature] = st.number_input(
                        feature,
                        value=default_val,
                        min_value=min_val,
                        max_value=max_val,
                        key=f"input_{feature}"
                    )
        
        st.divider()
        
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.markdown("### 📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 3rem; border-radius: 15px; text-align: center; color: white;">
                    <h2 style="margin: 0; color: white;">Predicted {target_col}</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0; color: white;">{prediction:,.2f}</h1>
                    <p style="opacity: 0.9; color: white;">Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Show input summary
            st.markdown("### 📋 Input Summary")
            input_summary = pd.DataFrame({
                'Feature': list(input_data.keys()),
                'Value': [f"{v:,.2f}" for v in input_data.values()]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("📊 Performance Reports")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")
    else:
        # Generate report
        report = f"""
# Predictive Analytics Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {model_type}
**Dataset:** {dataset_choice if data_source == 'Sample Dataset' else 'Custom Upload'}

## Model Performance

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| R² Score | {st.session_state['metrics']['train_r2']:.4f} | {st.session_state['metrics']['test_r2']:.4f} |
| RMSE | {st.session_state['metrics']['train_rmse']:,.2f} | {st.session_state['metrics']['test_rmse']:,.2f} |
| MAE | {st.session_state['metrics']['train_mae']:,.2f} | {st.session_state['metrics']['test_mae']:,.2f} |

## Configuration
- **Algorithm:** {model_type}
- **Test Split:** {test_size}%
- **Training Samples:** {len(X_train):,}
- **Test Samples:** {len(X_test):,}
- **Features:** {len(st.session_state['feature_names'])}

## Interpretation
- The model achieves an R² score of {st.session_state['metrics']['test_r2']:.3f} on test data
- Average prediction error (MAE): {st.session_state['metrics']['test_mae']:,.2f}
- Root mean squared error: {st.session_state['metrics']['test_rmse']:,.2f}
"""
        
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        st.divider()
        
        # Export predictions
        st.markdown("### 📤 Export Predictions")
        
        pred_data = st.session_state['predictions']
        export_df = pd.DataFrame({
            'Actual': pred_data['y_test'],
            'Predicted': pred_data['y_pred'],
            'Error': pred_data['y_test'] - pred_data['y_pred'],
            'Absolute_Error': np.abs(pred_data['y_test'] - pred_data['y_pred'])
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 2rem;">
    <p>Built with Streamlit • Powered by scikit-learn • © 2024 Predictive Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)
