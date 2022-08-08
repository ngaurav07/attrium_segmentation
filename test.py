import streamlit as st
from attrium import extract
from PIL import Image
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
placeholder_image = st.empty()
for uploaded_file in uploaded_files:
     extract(uploaded_file.name, st, placeholder_image)
