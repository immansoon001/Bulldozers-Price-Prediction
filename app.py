import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Bulldozer Price Prediction",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE FUNCTIONS FOR MODEL AND FEATURES
# ============================================================================
@st.cache_resource
def load_model(model_path="model.pkl"):
    """Load pre-trained RandomForest model."""
    possible_paths = [
        model_path,
        os.path.join("model", "model.pkl"),
        os.path.join(Path.cwd(), "model", "model.pkl")
    ]
    
    model_found = None
    for path in possible_paths:
        if os.path.exists(path):
            model_found = path
            break
    
    if model_found is None:
        st.error("❌ **Model file not found**")
        st.warning(f"""
        Please ensure model.pkl exists in the working directory.
        Run the Jupyter notebook: `Bulldozer-Price-Predictions.ipynb`
        """)
        return None
    
    try:
        model = joblib.load(model_found)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_features(features_path="features.pkl"):
    """Load feature names from saved file."""
    possible_paths = [
        features_path,
        os.path.join("model", "features.pkl"),
        os.path.join(Path.cwd(), "model", "features.pkl")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                features = joblib.load(path)
                return features.tolist() if hasattr(features, 'tolist') else list(features)
            except Exception as e:
                st.warning(f"Could not load features from {path}: {str(e)}")
    
    return None

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def preprocess_input(input_dict, features_list):
    """
    Convert raw input to model-ready format.
    - Convert categorical columns to numeric codes
    - Create feature vector with all required columns
    - Handle missing values
    """
    df_input = pd.DataFrame([input_dict])
    
    # Known categorical columns that need encoding
    categorical_cols = [
        'state', 'ProductSize', 'UsageBand', 'fiModelDesc', 
        'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor',
        'fiProductClassDesc', 'ProductGroup', 'ProductGroupDesc',
        'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
        'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
        'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
        'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
        'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
        'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
        'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
        'Travel_Controls', 'Differential_Type', 'Steering_Controls'
    ]
    
    # Encode categorical columns
    for col in categorical_cols:
        if col in df_input.columns:
            df_input[col] = pd.Categorical(df_input[col]).codes + 1
    
    # Create output dataframe with all features
    df_output = pd.DataFrame(0, index=[0], columns=features_list)
    
    # Fill in available features
    for col in df_input.columns:
        if col in features_list:
            df_output[col] = df_input[col].values[0]
    
    # Ensure correct column order
    df_output = df_output[features_list]
    
    return df_output

# ============================================================================
# APP TITLE AND DESCRIPTION
# ============================================================================
st.title("🚜 Bulldozer Price Prediction")
st.markdown("""
Predict the sale price of a bulldozer using machine learning.
Fill in the bulldozer specifications below to get an instant price estimate.
""")

# ============================================================================
# SIDEBAR - INSTRUCTIONS AND DIAGNOSTICS
# ============================================================================
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    ### How to use:
    1. Fill in the bulldozer specifications
    2. Click the **Predict Price** button
    3. View the predicted price and estimate range
    
    ### Key Features:
    - **Year Made**: Manufacturing year
    - **Machine Hours**: Total operating hours
    - **Usage Band**: Low/Medium/High intensity use
    - **Product Size**: Physical size category
    - **State**: Location
    - **Sale Date**: Year, month, and day of sale
    """)
    
    with st.expander("🔧 Model Diagnostics"):
        st.markdown("**Model Status:**")
        model_status = "✓ Loaded" if os.path.exists("model.pkl") else "✗ Not Found"
        features_status = "✓ Loaded" if os.path.exists("features.pkl") else "✗ Not Found"
        
        st.write(f"Model: {model_status}")
        st.write(f"Features: {features_status}")
        st.write(f"**Working Dir:** `{Path.cwd()}`")

# ============================================================================
# LOAD MODEL AND FEATURES
# ============================================================================
model = load_model()
feature_names = load_features()

if model is None or feature_names is None:
    st.error("⚠️ Cannot proceed without model and features. Please check the diagnostics in the sidebar.")
    st.stop()

st.success(f"✓ Model loaded successfully ({model.n_features_in_} features)")

# ============================================================================
# USER INPUT SECTION
# ============================================================================
st.divider()
st.subheader("📋 Bulldozer Specifications")

# Create organized input sections with columns
tab1, tab2, tab3 = st.tabs(["Basic Info", "Technical Specs", "Sale Details"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        year_made = st.number_input(
            "Year Made",
            min_value=1960,
            max_value=2030,
            value=2010,
            step=1,
            help="Manufacturing year of the bulldozer"
        )
    
    with col2:
        machine_hours = st.number_input(
            "Machine Hours (Current Meter)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=100,
            help="Total operating hours on the machine"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        state = st.selectbox(
            "State",
            options=sorted(['Alabama', 'Arizona', 'Arkansas', 'California', 
                          'Colorado', 'Florida', 'Georgia', 'Illinois', 
                          'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
                          'Michigan', 'Minnesota', 'Missouri', 'Mississippi',
                          'North Carolina', 'New York', 'Ohio', 'Oklahoma',
                          'Oregon', 'Pennsylvania', 'Texas', 'Utah', 'Washington',
                          'Wisconsin', 'Other']),
            help="State where the bulldozer will be/was auctioned"
        )
    
    with col4:
        usage_band = st.selectbox(
            "Usage Band",
            options=["Low", "Medium", "High"],
            help="Typical usage intensity (Low/Medium/High)"
        )

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        product_size = st.selectbox(
            "Product Size",
            options=["Mini", "Compact", "Small", "Medium", "Large / Medium", "Large"],
            help="Physical size category of the bulldozer"
        )
    
    with col2:
        product_group = st.selectbox(
            "Product Group",
            options=["WL", "SSL", "TEX", "BL", "TTT", "MG"],
            help="Equipment type (WL=Wheel Loader, SSL=Skid Steer Loader, etc.)"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        drive_system = st.selectbox(
            "Drive System",
            options=["No", "Two Wheel Drive", "Four Wheel Drive", "All Wheel Drive", "None or Unspecified"],
            help="Type of drive system"
        )
    
    with col4:
        transmission = st.selectbox(
            "Transmission",
            options=["Standard", "Powershift", "Powershuttle", "Hydrostatic", "None or Unspecified"],
            help="Transmission type"
        )

with tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sale_year = st.number_input(
            "Sale Year",
            min_value=2000,
            max_value=2030,
            value=datetime.now().year,
            step=1,
            help="Year of sale"
        )
    
    with col2:
        sale_month = st.slider(
            "Sale Month",
            min_value=1,
            max_value=12,
            value=datetime.now().month,
            help="Month of sale (1=Jan, 12=Dec)"
        )
    
    with col3:
        sale_day = st.slider(
            "Sale Day",
            min_value=1,
            max_value=31,
            value=min(datetime.now().day, 28),
            help="Day of sale"
        )

# ============================================================================
# PREDICTION SECTION
# ============================================================================
st.divider()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    predict_button = st.button("🔮 Predict Price", use_container_width=True, type="primary")

with col2:
    show_debug = st.checkbox("Debug Info", value=False)

# ============================================================================
# MAKE PREDICTION
# ============================================================================
if predict_button:
    try:
        # Create input dictionary
        input_data = {
            'SalesID': 0,
            'MachineID': 0,
            'ModelID': 0,
            'datasource': 0,
            'auctioneerID': 0,
            'YearMade': year_made,
            'MachineHoursCurrentMeter': machine_hours,
            'UsageBand': usage_band,
            'fiModelDesc': 'Unknown',
            'fiBaseModel': 'Unknown',
            'fiSecondaryDesc': 'Unknown',
            'fiModelSeries': 'Unknown',
            'fiModelDescriptor': 'Unknown',
            'ProductSize': product_size,
            'fiProductClassDesc': 'Unknown',
            'state': state,
            'ProductGroup': product_group,
            'ProductGroupDesc': 'Unknown',
            'Drive_System': drive_system,
            'Enclosure': 'Unknown',
            'Forks': 0,
            'Pad_Type': 'Unknown',
            'Ride_Control': 'Unknown',
            'Stick': 'Unknown',
            'Transmission': transmission,
            'Turbocharged': 'Unknown',
            'Blade_Extension': 'Unknown',
            'Blade_Width': 'Unknown',
            'Enclosure_Type': 'Unknown',
            'Engine_Horsepower': 'Unknown',
            'Hydraulics': 'Unknown',
            'Pushblock': 'Unknown',
            'Ripper': 'Unknown',
            'Scarifier': 'Unknown',
            'Tip_Control': 'Unknown',
            'Tire_Size': 'Unknown',
            'Coupler': 'Unknown',
            'Coupler_System': 'Unknown',
            'Grouser_Tracks': 'Unknown',
            'Hydraulics_Flow': 'Unknown',
            'Track_Type': 'Unknown',
            'Undercarriage_Pad_Width': 'Unknown',
            'Stick_Length': 'Unknown',
            'Thumb': 'Unknown',
            'Pattern_Changer': 'Unknown',
            'Grouser_Type': 'Unknown',
            'Backhoe_Mounting': 'Unknown',
            'Blade_Type': 'Unknown',
            'Travel_Controls': 'Unknown',
            'Differential_Type': 'Unknown',
            'Steering_Controls': 'Unknown',
            'saleYear': sale_year,
            'saleMonth': sale_month,
            'saleDay': sale_day,
            'saleDayOfWeek': pd.Timestamp(year=sale_year, month=sale_month, day=sale_day).dayofweek,
            'saleDayOfYear': pd.Timestamp(year=sale_year, month=sale_month, day=sale_day).dayofyear,
        }
        
        # Show debug info if requested
        if show_debug:
            with st.expander("📊 Input Data", expanded=True):
                st.json(input_data)
        
        # Preprocess input
        input_processed = preprocess_input(input_data, feature_names)
        
        if show_debug:
            with st.expander("🔧 Processed Features", expanded=True):
                st.dataframe(input_processed.T, use_container_width=True)
        
        # Make prediction
        prediction = model.predict(input_processed)
        predicted_price = prediction[0]
        
        # Display results
        st.success("✓ Prediction Complete!")
        
        # Main result
        st.divider()
        col_pred, col_range = st.columns(2)
        
        with col_pred:
            st.metric(
                label="💰 Predicted Sale Price",
                value=f"${predicted_price:,.0f}",
                delta=None
            )
        
        with col_range:
            low_estimate = predicted_price * 0.80
            high_estimate = predicted_price * 1.20
            st.metric(
                label="📊 Estimated Range (±20%)",
                value=f"${low_estimate:,.0f} — ${high_estimate:,.0f}"
            )
        
        # Input summary
        st.divider()
        st.subheader("📝 Input Summary")
        
        summary_data = {
            'Specification': [
                'Year Made',
                'Machine Hours',
                'Usage Band',
                'Product Size',
                'State',
                'Product Group',
                'Drive System',
                'Transmission',
                'Sale Date'
            ],
            'Value': [
                str(year_made),
                f"{machine_hours:,} hours",
                usage_band,
                product_size,
                state,
                product_group,
                drive_system,
                transmission,
                f"{sale_month}/{sale_day}/{sale_year}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        if show_debug:
            st.write("Full error details:")
            st.code(str(e))

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Bulldozer Price Prediction Model**  
*Powered by Random Forest Regressor*  
Dataset: Blue Book for Bulldozers  
Model Features: 102 | Training Samples: 400,000+
""")