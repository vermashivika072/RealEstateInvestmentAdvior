import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
clf = joblib.load("models/good_investment_clf.joblib")
reg = joblib.load("models/future_price_reg.joblib")

# Load locality median
locality_df = pd.read_csv("models/locality_medians.csv")

st.title("🏠 Real Estate Investment Advisor")
st.write("Check if a property is a good investment and see its 5-year predicted price.")

# Layout: Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price (in lakhs)", min_value=0.0, value=50.0)
    size = st.number_input("Size (in sqft)", min_value=1.0, value=1000.0)
    bhk = st.number_input("BHK", min_value=1, value=2)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2100, value=2015)

with col2:
    amenities = st.text_input("Amenities (comma separated)", value="Gym,Pool")
    owner_type = st.selectbox("Owner Type", ["Builder", "Individual", "Agent", "Unknown"])
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House", "Unknown"])
    furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi", "Fully", "Unknown"])
    availability_status = st.selectbox("Availability Status", ["Available", "Under Construction", "Sold", "Unknown"])
    city = st.text_input("City", value="Mumbai")
    locality = st.text_input("Locality", value="")

# Function to create input row
def create_input_row():
    ppsq = price / max(size,1)
    amenities_count = len([a for a in amenities.split(",") if a.strip() != ""])
    if locality in list(locality_df['Locality']):
        median_ppsqft = float(locality_df.loc[locality_df['Locality']==locality,'locality_median_ppsqft'].median())
    else:
        median_ppsqft = float(locality_df['locality_median_ppsqft'].median())
    price_vs_median = ppsq - median_ppsqft
    age = 2025 - int(year_built)
    row = pd.DataFrame([{
        'Price_in_Lakhs': price,
        'Size_in_SqFt': size,
        'Price_per_SqFt': ppsq,
        'Age_of_Property': age,
        'BHK': bhk,
        'Amenities_Count': amenities_count,
        'Price_vs_Median_locality': price_vs_median,
        'Owner_Type': owner_type,
        'Property_Type': property_type,
        'Furnished_Status': furnished_status,
        'Availability_Status': availability_status,
        'City': city
    }])
    return row, price_vs_median, ppsq, median_ppsqft

# Predict button
if st.button("Predict"):
    row, price_vs_median, ppsq, median_ppsqft = create_input_row()
    is_good = clf.predict(row)[0]
    future_price = reg.predict(row)[0]

    # Color-coded result
    if is_good == 1:
        st.markdown(f"<h2 style='color:green;'>✅ Good Investment</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:red;'>❌ Not a Good Investment</h2>", unsafe_allow_html=True)

    st.markdown(f"💰 **Predicted Price after 5 years:** {future_price:.2f} lakhs")

    # Plot: Price per SqFt vs Median
    st.subheader("Price Comparison")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(['Median Locality Price','Property Price per SqFt'], [median_ppsqft, ppsq], color=['gray','blue'])
    ax.set_ylabel("Price per SqFt (lakhs/sqft)")
    st.pyplot(fig)
