import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt

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
        color: #ffffff !important;
        border: 1px solid #2c3e50 !important;
        font-weight: 400 !important;
    }
    .stButton>button:hover {
        background-color: #5be4c5 !important;
    }
    .stMarkdown, h1, h2, h3, p, span, label, .stSlider [data-baseweb="caption"] {
        color: #2c3e50 !important;
    }
    /* Ensure help text under sliders is visible */
    .stSlider [data-testid="stText"] {
        color: #2c3e50 !important;
    }
    /* Style for form submit button */
    .stButton>button[type="submit"], .stButton button:has([data-testid="stFormSubmitButton"]) {
        color: #ffffff !important;
        font-weight: 400 !important;
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
    /* Ensure slider values are visible */
    .stSlider label, 
    .stSlider [data-testid="stMarkdownContainer"] p,
    .stSlider [data-baseweb="caption"],
    .stSlider span,
    div[data-baseweb="slider"] div[role="slider"] + div,
    .stMarkdown div[data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
    }
    /* Style for numeric input values and range labels */
    input[type="number"],
    .stSlider [data-baseweb="typography"],
    .stSlider [data-testid="stWidgetLabel"],
    .stSlider span,
    div[data-baseweb="slider"] div > span {
        color: #2c3e50 !important;
    }
    /* Specifically target the min/max range values */
    div[data-baseweb="slider"] span {
        color: #2c3e50 !important;
    }
    /* Style the predict button text */
    .stButton>button[type="submit"] {
        color: #4addbe  !important;
    }
    /* Additional styling for slider range values */
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        color: #2c3e50 !important;
    }
    .stSlider div[data-baseweb="slider"] div[role="slider"] + div {
        color: #2c3e50 !important;
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
    model = xgb.XGBClassifier(n_estimators=20, eval_metric='logloss')
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, 
             eval_set=eval_set,
             verbose=False)
    
    # Get evaluation results
    eval_result = model.evals_result()
    
    # Generate predictions for test set
    test_predictions = model.predict(tester.drop(columns=["fail"]))
    
    return model, X.columns, eval_result, tester.fail, test_predictions

def plot_training_metrics(eval_result):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    training_rounds = range(len(eval_result['validation_0']['logloss']))
    
    plt.scatter(x=training_rounds, 
               y=eval_result['validation_0']['logloss'], 
               label='train',
               color='#4addbe')  # turquoise
    plt.scatter(x=training_rounds, 
               y=eval_result['validation_1']['logloss'], 
               label='val', 
               color='#2c3e50',  # charcoal
               alpha=0.6)
    
    plt.xlabel('Training Rounds')
    plt.ylabel('Log Loss')
    plt.title('Training and Validation Metrics')
    plt.legend()
    
    return fig

def format_classification_report(y_true, y_pred):
    """Format classification report as a styled table"""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Create formatted table
    html = """
    <table style='width:100%; border-collapse: collapse; margin: 10px 0;'>
        <tr style='border-bottom: 2px solid #36454F;'>
            <th style='text-align:left; padding:8px;'>Class</th>
            <th style='text-align:center; padding:8px;'>Precision</th>
            <th style='text-align:center; padding:8px;'>Recall</th>
            <th style='text-align:center; padding:8px;'>F1-score</th>
            <th style='text-align:center; padding:8px;'>Support</th>
        </tr>
    """
    
    # Add rows for each class
    for label in ['0', '1']:
        metrics = report_dict[label]
        html += f"""
        <tr style='border-bottom: 1px solid #ddd;'>
            <td style='padding:8px;'>{label}</td>
            <td style='text-align:center; padding:8px;'>{metrics['precision']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['recall']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['f1-score']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['support']}</td>
        </tr>
        """
    
    # Add summary rows
    for metric in ['macro avg', 'weighted avg']:
        metrics = report_dict[metric]
        html += f"""
        <tr style='border-top: 1px solid #36454F;'>
            <td style='padding:8px;'>{metric}</td>
            <td style='text-align:center; padding:8px;'>{metrics['precision']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['recall']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['f1-score']:.2f}</td>
            <td style='text-align:center; padding:8px;'>{metrics['support']}</td>
        </tr>
        """
    
    html += "</table>"
    return html

def save_model(model):
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def main():
    # Initialize session state variables if they don't exist
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'eval_result' not in st.session_state:
        st.session_state.eval_result = None
    if 'test_actual' not in st.session_state:
        st.session_state.test_actual = None
    if 'test_pred' not in st.session_state:
        st.session_state.test_pred = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = None

    st.markdown("""
        <style>
        .dataframe {color: #36454F !important;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #36454F;'>ML Model Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #36454F;'>Make predictions with our trained model</p>", unsafe_allow_html=True)
    
    # Training section
    st.markdown("<h2 style='color: #36454F;'>Model Training</h2>", unsafe_allow_html=True)
    
    if st.button("Train Model"):
        with st.spinner("Training in progress... Hold on to your kippah!"):
            try:
                model, features, eval_result, test_actual, test_pred = train_model()
                st.session_state.model = model
                st.session_state.feature_columns = features
                st.session_state.eval_result = eval_result
                st.session_state.test_actual = test_actual
                st.session_state.test_pred = test_pred
                save_model(model)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

    # Display visualizations if we have training results
    if st.session_state.eval_result is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("This plot shows that both the training set and validation set are accurately predicting on unseen data with consistent logloss to prevent over-fitting.")
            fig = plot_training_metrics(st.session_state.eval_result)
            st.pyplot(fig)
        
        with col2:
            st.markdown("The model is performing quite well, especially with predicting model failures (class 1) without over-fitting as evidenced by the plot to the left.")
            st.markdown("### Classification Report")
            st.image("cr.png", use_container_width=True)

    # Prediction section
    st.markdown("<h2 style='color: #2c3e50;'>Make Predictions</h2>", unsafe_allow_html=True)
    st.markdown("""
    A chemical engineering system is made up of different machines that work together as part of a larger setup. 
    Each machine has a sensor to monitor how it's performing. Apart from the machine settings, external factors 
    like air quality can also impact how long the system lasts. You can adjust the settings and predict how 
    the system will perform based on those changes.
    """)
    
    try:
        if st.session_state.feature_columns is not None:
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
                    # Make prediction using session state model
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state.model.predict(input_df)
                    probability = st.session_state.model.predict_proba(input_df)
                    
                    # Display results with styled header
                    st.markdown("<h3 style='color: #4addbe'>Prediction Results</h3>", unsafe_allow_html=True)
                    result = 'Fail' if prediction[0] == 1 else 'Pass'
                    
                    # Get probability for the predicted class
                    prob = probability[0][1] if prediction[0] == 1 else probability[0][0]
                    
                    # Colored box based on prediction
                    if result == 'Pass':
                        st.success(f"Prediction: {result} (Confidence: {prob:.2%})")
                    else:
                        st.error(f"Prediction: {result} (Confidence: {prob:.2%})")
                    
        else:
            st.info("Please train the model first before making predictions!")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()