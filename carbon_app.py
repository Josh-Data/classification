import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go

# Set page config for dark theme
st.set_page_config(
    page_title="Steel Industry CO2 Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

def add_pd_times(df):
    """Add time-based features to dataframe"""
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    return df

def train_model(df):
    """Train the XGBoost model"""
    length = df.shape[0]
    main = int(length * 0.8)
    trainer = df[:main]
    tester = df[main:]
    
    X = trainer.drop(columns=["CO2(tCO2)"])
    y = trainer["CO2(tCO2)"]
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42, shuffle=False)
    
    model = xgb.XGBRegressor(n_estimators=25, learning_rate=0.1, max_depth=7, subsample=1.0)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    
    return model, tester, X_train.columns

def plot_predictions(tester, model):
    """Create prediction plot using plotly"""
    X_test = tester.drop(columns=["CO2(tCO2)"])
    predictions = model.predict(X_test)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester["CO2(tCO2)"],
        name="Actual",
        line=dict(color="#00ffcc", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=predictions,
        name="Predicted",
        line=dict(color="#ff00ff", width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="CO2 Emissions: Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="CO2 Emissions (tCO2)",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#00ffcc"),
        showlegend=True,
        legend=dict(
            bgcolor="#111111",
            bordercolor="#00ffcc"
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#00ffcc20'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#00ffcc20'
        )
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Create feature importance plot"""
    importance = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=features_df['Importance'],
        y=features_df['Feature'],
        orientation='h',
        marker=dict(color="#00ffcc")
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#00ffcc"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#00ffcc20'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#00ffcc20'
        )
    )
    
    return fig

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    # Load and process data
    try:
        df = pd.read_csv("Steel_industry_data.csv", index_col="date")
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        df = add_pd_times(df)
        df = df.drop(columns=["Day_of_week"])
        
        # Training section
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training in progress... 🔄"):
                model, test_data, feature_names = train_model(df)
                
                # Save model and test data in session state
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                
                st.success("Model trained successfully! 🎉")
        
        # Visualization section
        if 'model' in st.session_state:
            st.header("Model Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predictions vs Actual Values")
                pred_fig = plot_predictions(st.session_state['test_data'], 
                                         st.session_state['model'])
                st.plotly_chart(pred_fig, use_container_width=True)
            
            with col2:
                st.subheader("Feature Importance")
                imp_fig = plot_feature_importance(st.session_state['model'],
                                               st.session_state['feature_names'])
                st.plotly_chart(imp_fig, use_container_width=True)
            
            # Add metrics
            predictions = st.session_state['model'].predict(
                st.session_state['test_data'].drop(columns=["CO2(tCO2)"])
            )
            actual = st.session_state['test_data']["CO2(tCO2)"]
            mse = np.mean((predictions - actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actual))
            
            st.header("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
            col3.metric("Mean Absolute Error", f"{mae:.2f}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the data file is in the correct location and format.")

if __name__ == "__main__":
    main()