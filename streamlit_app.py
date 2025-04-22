import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

# Load pre-trained model, encoder, and scaler
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI configuration
st.set_page_config(page_title="Prediction Cancelling Hotel", page_icon="🏨", layout="centered")
st.markdown("<h1 style='text-align: center;'>Prediction On Cancelling Booking Hotel 🏨</h1>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("Dataset_B_hotel.csv")

df = load_data()

st.markdown("<h4 style='text-align: center;'>Here's an example of the Raw Hotel Dataset that is given in the question!</h4>", unsafe_allow_html=True)
with st.expander("📂 Raw Data "):
    st.dataframe(df.head(10))

# User input form for prediction
st.markdown("<h4 style='text-align: center;'>Input The Data you want to predict 🔮!</h4>", unsafe_allow_html=True)
with st.form("booking_form"):
    st.subheader("📋 Informasi Pemesanan")
    
    col1, col2 = st.columns(2)
    with col1:
        no_of_adults = st.number_input("👨‍👩‍👧‍👦 Jumlah Dewasa", min_value=0, value=2)
        no_of_children = st.number_input("🧒 Jumlah Anak", min_value=0, value=0)
        no_of_weekend_nights = st.number_input("🌙 Malam Akhir Pekan", min_value=0, value=1)
        no_of_week_nights = st.number_input("📆 Malam Hari Kerja", min_value=0, value=2)
        meal_plan_choice = st.selectbox("🍽️ Meal Plan", ["No Meal", "Meal Type 1", "Meal Type 2", "Meal Type 3"])
        required_car_parking_space = st.selectbox("🚗 Butuh Parkir?", options=[0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        room_type_choice = st.selectbox("🛏️ Tipe Kamar", ["Tipe 1 ", "Tipe 2 ", "Tipe 3 ", "Tipe 4 ", "Tipe 5 ", "Tipe 6 ", "Tipe 7 "])
        lead_time = st.number_input("⏳ Lead Time (hari sebelum check-in)", min_value=0, value=30)

    with col2:
        arrival_year = st.number_input("📅 Tahun Kedatangan", min_value=2020, value=2024)
        arrival_month = st.slider("🗓️ Bulan Kedatangan", 1, 12, 5)
        arrival_date = st.slider("📆 Tanggal Kedatangan", 1, 31, 15)
        market_choice = st.selectbox("🧭 Segmentasi Pasar", ["Online", "Offline", "Agen Perjalanan", "Perusahaan", "Langganan", "Lainnya"])
        repeated_guest = st.selectbox("🔁 Tamu Berulang?", options=[0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        no_of_previous_cancellations = st.number_input("❌ Cancel Sebelumnya", min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("✅ Booking Berhasil Sebelumnya", min_value=0, value=1)
        avg_price_per_room = st.number_input("💸 Harga Rata-Rata Kamar", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("📝 Jumlah Request Khusus", min_value=0, value=0)

    st.markdown("")
    submit = st.form_submit_button("🔮 Prediksi Sekarang")

if submit:
    with st.spinner("⏳ Sedang memproses prediksi..."):
        input_data = pd.DataFrame([[
            no_of_adults, no_of_children, no_of_weekend_nights,
            no_of_week_nights, meal_plan_choice, required_car_parking_space,
            room_type_choice, lead_time, arrival_year, arrival_month,
            arrival_date, market_choice, repeated_guest,
            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
            avg_price_per_room, no_of_special_requests
        ]], columns=[
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'meal_plan_choice', 'required_car_parking_space',
            'room_type_choice', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'market_choice', 'repeated_guest',
            'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ])

        # Apply encoder and scaler to the input data
       # Assuming you have loaded the encoder and scaler as encoder.pkl and scaler.pkl
        with open('encoder.pkl', 'rb') as file:
            encoder = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        # Define the categorical columns
        categorical_columns = ['meal_plan_choice', 'room_type_choice', 'market_choice']
        
        # Perform one-hot encoding on categorical columns
        encoded_input = encoder.transform(input_data[categorical_columns])
        
        # Convert the one-hot encoded data back to a DataFrame for easier manipulation
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_columns))
        
        # Now we concatenate the encoded data with the other non-categorical columns
        non_categorical_columns = input_data.drop(columns=categorical_columns)
        final_input_data = pd.concat([non_categorical_columns, encoded_df], axis=1)
        
        # Now scale the numeric columns if needed
        final_input_data_scaled = scaler.transform(final_input_data)
        
        # Pass the scaled data to the model for prediction
        prediction = model.predict(final_input_data_scaled)
        prediction_prob = model.predict_proba(final_input_data_scaled)[:, 1]
        
        # Output the prediction
        st.write(f"Probability that the booking is canceled: {prediction_prob[0]:.4f}")
        st.write(f"Probability that the booking is not canceled: {1 - prediction_prob[0]:.4f}")


        st.subheader("📝 Data Input By User")
        st.dataframe(input_data)

        st.subheader("📊 Probability Prediction")
        st.write(f"Probability that the booking canceled: {prediction_prob[0]:.4f}")
        st.write(f"Probability that the booking not canceled {1 - prediction_prob[0]:.4f}")

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

        st.markdown("### 🧠 Feature Contribution to the Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_data)

        shap_df = pd.DataFrame({
            'Feature': transformed_data.columns,
            'SHAP Value': shap_values[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette="coolwarm", ax=ax2)
        ax2.set_title("Feature Impact on This Prediction")
        st.pyplot(fig2)

        hasil = "✅ Booking Not Canceled!" if prediction[0] == 1 else "❌ Booking Canceled!"
        st.markdown("---")
        st.subheader("🎯 Prediction Result : ")
        if prediction[0] == 1:
            st.success(hasil)
        else:
            st.error(hasil)

