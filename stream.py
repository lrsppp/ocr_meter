import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

from ocr_meter.const import API_URL


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


st.title("OCR Meter")
st.markdown(
    "### Instructions\n"
    "1. Upload a PNG file by using the file uploader.\n"
    "2. View the result in the table below.\n"
    "3. Export CSV"
)

# Initialize Table
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["File Name", "Digit"])

table_placeholder = st.empty()
table_placeholder.data_editor(st.session_state.df)

csv = convert_df(st.session_state.df)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
)

# Upload Image
uploaded_file = st.file_uploader("Upload Images by Drag & Drop", type="png")
if uploaded_file is not None:
    st.subheader("Image")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Read image
    file_content = uploaded_file.read()
    data = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image_data = {"image_data": image.tolist()}

    # Send to API
    response = requests.post(API_URL, json=image_data)
    if response.status_code == 200:
        result = response.json()
        st.session_state.df = pd.concat(
            [
                st.session_state.df,
                pd.DataFrame(
                    {"File Name": [uploaded_file.name], "Digit": [result["digit"]]}
                ),
            ]
        )
        table_placeholder.empty()
        table_placeholder.data_editor(st.session_state.df)
    else:
        st.error("Error calling OCR API")
