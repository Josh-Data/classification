import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split as tts
import pickle

# Configure the page layout
st.set_page_config(
    page_title="ML Model Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .css-1d391kg {
        background-color: white;
    }
    .stButton>button {
        background-color: #4addbe;
        color: white;
        border: 1px solid #2c3e50 !important;
    }
    .stMarkdown, h1, h2, h3, p, span, label {
        color: #2c3e50 !important;
    }
    /* Style for slider - new streamlit class names */
    .st-emotion-cache-1y4p8pa {
        width: 100%;
    }
    .st-emotion-cache-1y4p8pa .stSlider > div > div > div {
        background-color: #4addbe !important;
    }
    .st-emotion-cache-1y4p8pa .stSlider > div > div > div > div {
        background-color: #4addbe !important;
    }
    </style>
""", unsafe_allow_html=True)

# Define default values based on median (50%) statistics
DEFAULT_VALUES = {
    'footfall': 22.0,
    'tempMode': 3.0,
    'AQ': 4.0,
    'USS': 3.0,
    'CS': 6.0,
    'VOC': 2.0,
    'RP': 4.0,
    'IP': 4.0,
    'Temperature': 44.0
}

# Define min-max values for sliders
VALUE_RANGES = {
    'footfall': (0.0, 7300.0),
    'tempMode': (0.0, 7.0),
    'AQ': (1.0, 7.0),
    'USS': (1.0, 7.0),
    'CS': (1.0, 7.0),
    'VOC': (0.0, 6.0),
    'RP': (1.0, 7.0),
    'IP': (1.0, 7.0),
    'Temperature': (19.0, 91.0)
}

def train_model():
    # Load the dataset
    df = pd.read_csv("data.csv")
    
    # Prepare the training and testing data
    length = df.shape[0]
    main = int(length * 0.8)
    trainer = df.iloc[:main]
    tester = df.iloc[main:]
    X = trainer.drop(columns=["fail"])
    y = trainer.fail
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42)
    
    # Train the model
    model = xgb.XGBClassifier(n_estimators=20)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    
    return model, X.columns

def save_model(model):
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def main():
    # Custom title styling
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>ML Model Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #2c3e50;'>Make predictions with our trained model</p>", unsafe_allow_html=True)
    
    # Store the feature columns globally after first training
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = None
    
    # Training section with styled header
    st.markdown("<h2 style='color: #2c3e50;'>Model Training</h2>", unsafe_allow_html=True)
    if st.button("Train Model"):
        with st.spinner("Training in progress... Hold on to your kippah!"):
            try:
                model, features = train_model()
                st.session_state.feature_columns = features
                save_model(model)
                st.success("Model trained successfully! Mazel tov! 🎉")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    # Prediction section with styled header
    st.markdown("<h2 style='color: #2c3e50;'>Make Predictions</h2>", unsafe_allow_html=True)
    
    try:
        if st.session_state.feature_columns is not None:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Create a form for user input
            with st.form("prediction_form"):
                st.markdown("<h3 style='color: #2c3e50;'>Enter values for prediction</h3>", unsafe_allow_html=True)
                
                # Create input fields for each feature with sliders
                input_data = {}
                
                # Create two columns for a better layout
                col1, col2 = st.columns(2)
                
                features = list(DEFAULT_VALUES.keys())
                mid_point = len(features) // 2
                
                # First column
                with col1:
                    for feature in features[:mid_point]:
                        min_val, max_val = VALUE_RANGES[feature]
                        input_data[feature] = st.slider(
                            f"{feature}",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(DEFAULT_VALUES[feature]),
                            step=0.1 if feature in ['footfall', 'Temperature'] else 1.0,
                            help=f"Range: {min_val} to {max_val}"
                        )
                
                # Second column
                with col2:
                    for feature in features[mid_point:]:
                        min_val, max_val = VALUE_RANGES[feature]
                        input_data[feature] = st.slider(
                            f"{feature}",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(DEFAULT_VALUES[feature]),
                            step=0.1 if feature in ['footfall', 'Temperature'] else 1.0,
                            help=f"Range: {min_val} to {max_val}"
                        )
                
                submitted = st.form_submit_button("Predict")
                if submitted:
                    # Make prediction
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    probability = model.predict_proba(input_df)
                    
                    # Display results with styled header
                    st.markdown("<h3 style='color: #2c3e50;'>Prediction Results</h3>", unsafe_allow_html=True)
                    result = 'Fail' if prediction[0] == 1 else 'Pass'
                    prob = probability[0][1]
                    
                    # Colored box based on prediction
                    if result == 'Pass':
                        st.success(f"Prediction: {result} (Probability: {prob:.2%})")
                    else:
                        st.error(f"Prediction: {result} (Probability: {prob:.2%})")
                    
        else:
            st.info("Please train the model first before making predictions!")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()