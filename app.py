import streamlit as st
import numpy as np
import pandas as pd
import os

from train import train_models
from utils import hybrid

# Page setup
st.set_page_config(page_title="Ad Recommendation", layout="wide")

# 🎨 UI Styling
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
h1, h2, h3 {color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 AI-Based Ad Recommendation System")
st.markdown("### Personalized Ads using Machine Learning + Hybrid Recommendation")
st.info("💡 This system recommends ads based on your activity and interests")
st.markdown("---")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
- AI-based Ad Recommendation  
- Uses ML (LR, RF, XGBoost)  
- Hybrid Filtering System  
- User Behavior Analysis  
""")

# Train models
lr, rf, xgb_model, le, acc = train_models()

# Show accuracy
st.subheader("📊 Model Performance")
st.write(acc)

# -----------------------------
# USER INPUT (CLEAN UI)
# -----------------------------
st.subheader("🎯 Tell us about yourself")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Your Age", 18, 60, 25)
    gender = st.selectbox("Select Gender", ["M", "F"])

with col2:
    engagement = st.selectbox(
        "How active are you online?",
        ["Low (rarely browse)", "Medium (sometimes active)", "High (very active)"]
    )

    interest = st.selectbox(
        "What are you interested in?",
        ["Tech", "Fashion", "Food"]
    )

# Convert engagement → numeric (for ML model)
if "Low" in engagement:
    time_spent = 2
elif "Medium" in engagement:
    time_spent = 7
else:
    time_spent = 12

# Optional user tracking
user_id = st.text_input("User ID (optional for tracking)")

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🚀 Show My Ads"):
    g = le.transform([gender])[0]
    user = np.array([[age, g, time_spent]])

    prob = xgb_model.predict_proba(user)[0][1]

    st.subheader("📈 Your Engagement Score")
    st.metric("Click Probability", round(prob, 2))

    # Hybrid recommendation
    ads = hybrid(prob, interest)

    st.subheader("📢 Ads Recommended For You")

    if isinstance(ads, str):
        st.warning(ads)
    else:
        for _, row in ads.iterrows():
            st.success(f"🔥 {row['brand']} ({row['category']})")

    # Save user history safely
    if user_id:
        new_data = pd.DataFrame({
            "user_id": [user_id],
            "interest": [interest],
            "time_spent": [time_spent]
        })

        if not os.path.exists("users.csv"):
            new_data.to_csv("users.csv", index=False)
        else:
            new_data.to_csv("users.csv", mode='a', header=False, index=False)

        st.success("✅ Your activity has been saved!")

# Footer
st.markdown("---")
st.success("🎯 Smart Ads Powered by AI + Hybrid Recommendation")