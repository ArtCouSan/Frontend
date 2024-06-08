import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

st.set_page_config(page_title='Análise de Corais Profundos 🌊', layout='wide')

st.title("Analise de Corais Profundos 🌊")

model = joblib.load('./pickle/iris_classifier_model.pkl')
encoder = joblib.load('./pickle/encoder.pkl')
label_encoder = joblib.load('./pickle/label_encoder.pkl')
normalizer = joblib.load('./pickle/normalizer.pkl')

with st.form("prediction_form"):
    st.write("📝 Preencha os dados para predição:")

    col1, col2, col3 = st.columns(3)

    with col1:
        data_provider = st.text_input("Data Provider", value="NOAA, Alaska Fisheries Science Center")
        scientific_name = st.text_input("Scientific Name", value="Porifera")
        vernacular_name_category = st.text_input("Vernacular Name Category", value="gorgonian coral")
        taxon_rank = st.text_input("Taxon Rank", value="species")
        station = st.text_input("Station", value="14")

    with col2:
        latitude = st.number_input("Latitude", value=-23.547500, format="%.6f")
        longitude = st.number_input("Longitude", value=-46.636110, format="%.6f")
        depth_in_meters = st.number_input("Depth in Meters", value=10.5, format="%.2f")
        depth_method = st.text_input("Depth Method", value="reported")
        locality = st.text_input("Locality", value="OLYMPIC COAST")

    with col3:
        location_accuracy = st.number_input("Location Accuracy", value=100, min_value=0, max_value=10000, step=10)
        repository = st.text_input("Repository", value="NOAA, Alaska Fisheries Science Center")
        identification_qualifier = st.text_input("Identification Qualifier", value="ID by expert from video")
        sampling_equipment = st.text_input("Sampling Equipment", value="ROV")
        record_type = st.text_input("Record Type", value="video observation")
        date_year = st.number_input("Date Year", value=2024, format="%d")
        count_by_date = st.number_input("Count By Date", value=1, format="%d")

    submitted = st.form_submit_button("Predict 🚀")
    if submitted:
        input_df = pd.DataFrame({
            'DataProvider': [data_provider],
            'ScientificName': [scientific_name],
            'VernacularNameCategory': [vernacular_name_category],
            'TaxonRank': [taxon_rank],
            'Station': [station],
            'latitude': [latitude],
            'longitude': [longitude],
            'DepthInMeters': [depth_in_meters],
            'DepthMethod': [depth_method],
            'Locality': [locality],
            'LocationAccuracy': [location_accuracy],
            'Repository': [repository],
            'IdentificationQualifier': [identification_qualifier],
            'SamplingEquipment': [sampling_equipment],
            'RecordType': [record_type],
            'DateYear': [date_year],
            'CountByDate': [count_by_date]
        })

        for column, frequency_map in encoder.items():
            input_df[column] = input_df[column].map(frequency_map)


        # Normalizar os dados
        input_normalized = normalizer.transform(input_df)
        prediction = model.predict(input_normalized)
        result = label_encoder.inverse_transform(prediction)
        st.write("Resultado da Predição 🌟:", result[0])
