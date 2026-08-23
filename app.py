import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EthioVal - Real Estate Valuation", page_icon="🏠", layout="centered")

st.title("🏠 EthioVal: Addis Ababa Residential Property Price Estimator.")
st.write("Enter property details below to estimate market value in ETB.")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgboost_model.joblib')
    return preprocessor, model

preprocessor, model = load_artifacts()

st.header("Property Details")

subcity = st.selectbox("Subcity", [
    "Bole", "Yeka", "Nifas Silk-Lafto", "Kirkos", "Arada", 
    "Gullele", "Kolfe Keranio", "Akaky Kaliti", "Addis Ketema", "Lideta"
])
property_type = st.selectbox("Property Type", ["Condominium", "Apartment", "House", "Villa", "Commercial"])
listing_type = st.selectbox("Listing Type", ["sale", "rent"])
furnishing = st.selectbox("Furnishing", ["unfurnished", "furnished", "semi-furnished"])
num_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
num_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
size_sqm = st.number_input("Size (in m²)", min_value=10.0, max_value=2000.0, value=120.0, step=5.0)

if st.button("Estimate Price", type="primary"):
    input_data = pd.DataFrame([{
        'subcity': subcity,
        'property_type': property_type,
        'listing_type': listing_type,
        'furnishing': furnishing,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'size_sqm': size_sqm,
        'lat': 9.0105,
        'lng': 38.7612,
        'dist_meskel_square': 3.5
    }])
    
    try:
        X_trans = preprocessor.transform(input_data)
        predicted_price = model.predict(X_trans)[0]
        
        st.success(f"### Estimated Price: {predicted_price:,.2f} ETB")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
