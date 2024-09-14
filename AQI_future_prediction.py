import numpy as np
import pandas as pd
import streamlit as st
import base64
import joblib   # it is use for saving and loading models in machine learning and data 

st.set_page_config(layout="wide",
                   page_icon="🍃",page_title="AQI Future Prediction")

@st.cache_data()
def load_model():
    model=joblib.load("")

