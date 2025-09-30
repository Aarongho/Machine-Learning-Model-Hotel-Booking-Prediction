import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ===========================
# Load Model & Dataset
# ===========================
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@st.cache_data
def load_data():
    return pd.read_csv("Dataset_B_hotel.csv")

df = load_data()

# ===========================
# Mapping Dictionaries
# ===========================
meal_plan_map = {
    "No Meal": 0,
    "Meal Type 1": 1,
    "Meal Type 2": 2,
    "Meal Type 3": 3
}

market_segment_map = {
    "Online": 0,
    "Offline": 1,
    "Agen Perjalanan": 2,
    "Perusahaan": 3,
    "Langganan": 4,
    "Lainnya": 5
}

room_type_map = {
    "Tipe 1 ": 0,
    "Tipe 2 ": 1,
    "Tipe 3 ": 2,
    "Tipe 4 ": 3,
    "Tipe 5 ": 4,
    "Tipe 6 ": 5,
    "Tipe 7 ": 6
}

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Prediction Cancelling Hotel", page_icon="🏨", layout="centered")
st.markdown("<h1 style='text-align: center;'>Prediction On Cancelling Booking Hotel 🏨</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("<h4 style='text-align: center;'>Here's an example of the Raw Hotel Dataset</h4>", unsafe_allow_html=True)
with st.expander("📂 Raw Data "):
    st.dataframe(df.head(10))

st.markdown("<h4 style='text-align: center;'>Input Data to Predict 🔮</h4>", unsafe_allow_html=True)

# ===========================
# Input Form
# ===========================
with st.form("booking_form"):
    st.subheader("📋 Informasi Pemesanan")
    
    col1, col2 = st.columns(2)
    with col1:
        no_of_adults = st.number_input("👨‍👩‍👧‍👦 Jumlah Dewasa", min_value=0, value=2)
        no_of_children = st.number_input("🧒 Jumlah Anak", min_value=0, value=0)
        no_of_weekend_nights = st.number_input("🌙 Malam Akhir Pekan", min_value=0, value=1)
        no_of_week_nights = st.number_input("📆 Malam Hari Kerja", min_value=0, value=2)
        meal_plan_choice = st.selectbox("🍽️ Meal Plan", list(meal_plan_map.keys()))
        type_of_meal_plan = meal_plan_map[meal_plan_choice]
        required_car_parking_space = st.selectbox("🚗 Butuh Parkir?", options=[0,1], format_func=lambda x: "Ya" if x else "Tidak")
        room_type_choice = st.selectbox("🛏️ Tipe Kamar", list(room_type_map.keys()))
        room_type_reserved = room_type_map[room_type_choice]
        lead_time = st.number_input("⏳ Lead Time (hari sebelum check-in)", min_value=0, value=30)

    with col2:
        arrival_year = st.number_input("📅 Tahun Kedatangan", min_value=2020, value=2024)
        arrival_month = st.slider("🗓️ Bulan Kedatangan", 1, 12, 5)
        arrival_date = st.slider("📆 Tanggal Kedatangan", 1, 31, 15)
        market_choice = st.selectbox("🧭 Segmentasi Pasar", list(market_segment_map.keys()))
        market_segment_type = market_segment_map[market_choice]
        repeated_guest = st.selectbox("🔁 Tamu Berulang?", options=[0,1], format_func=lambda x: "Ya" if x else "Tidak")
        no_of_previous_cancellations = st.number_input("❌ Cancel Sebelumnya", min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("✅ Booking Berhasil Sebelumnya", min_value=0, value=1)
        avg_price_per_room = st.number_input("💸 Harga Rata-Rata Kamar", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("📝 Jumlah Request Khusus", min_value=0, value=0)

    submit = st.form_submit_button("🔮 Prediksi Sekarang")

# ===========================
# Prediction Logic
# ===========================
if submit:
    with st.spinner("⏳ Sedang memproses prediksi..."):
        input_data = pd.DataFrame([[
            no_of_adults, no_of_children, no_of_weekend_nights,
            no_of_week_nights, type_of_meal_plan, required_car_parking_space,
            room_type_reserved, lead_time, arrival_year, arrival_month,
            arrival_date, market_segment_type, repeated_guest,
            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
            avg_price_per_room, no_of_special_requests
        ]], columns=[
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
            'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'market_segment_type', 'repeated_guest',
            'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ])

        # Predict
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)[:, 1]

        # Show input
        st.subheader("📝 Data Input By User")
        st.dataframe(input_data)

        # Probability
        st.subheader("📊 Probability Prediction")
        st.write(f"Probability that the booking canceled: {prediction_prob[0]:.4f}")
        st.write(f"Probability that the booking not canceled: {1 - prediction_prob[0]:.4f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#FF6B6B', '#4CAF50']
        labels = ['Canceled', 'Not Canceled']
        probs = [prediction_prob[0], 1 - prediction_prob[0]]
        bars = ax.bar(labels, probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title("Probability of Booking Outcome")
        ax.set_ylabel("Probability")
        ax.bar_label(bars, fmt='%.2f', padding=3)
        st.pyplot(fig)

        # SHAP
        st.markdown("### 🧠 Feature Contribution to the Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        shap_df = pd.DataFrame({
            'Feature': input_data.columns,
            'SHAP Value': shap_values.values[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette="coolwarm", ax=ax2)
        ax2.set_title("Feature Impact on This Prediction")
        st.pyplot(fig2)

        # Final Result
        hasil = "✅ Booking Not Canceled!" if prediction[0] == 1 else "❌ Booking Canceled!"
        st.markdown("---")
        st.subheader("🎯 Prediction Result : ")
        if prediction[0] == 1:
            st.success(hasil)
        else:
            st.error(hasil)
