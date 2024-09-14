import streamlit as st

import base64

st.set_page_config(layout="wide",page_icon="🌍",page_title="AQI Awareness Hub")


@st.cache_data
def get_image_as_base64(file):
    with open(file,'rb') as f:
        data=f.read()
    return base64.b64encode(data).decode()

img=get_image_as_base64('template/image002.jpg')
st.text('');
st.text('');
st.text('');
st.text('');


st.markdown("""<h1><span data-id="s1" class="spans">AQI</span><span data-id="s2"> Awareness Hub 🌍</span></h1>""",
            unsafe_allow_html=True)

with open("template/title.css") as file:
    st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)


# for show background image
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"]{{
    background-image: url("data:image/png;base64,{img}");
    background-color: grey;
    background-size: cover;
    opacity: blur;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .transparent-iframe iframe {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True)

# st.components.v1.html(html_code,height=150)
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.link_button('Explore now!','/home')
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');

st.markdown('''
    This app is a deep dive analysis of the air quality parameters over the years in India. Also it allow user to compare them in different ways. The app tried to show 
    every dimension possible to understand the air quality in India.
            
    There are many hidden facts and relations that were found during the analysis of the data and which are vital to
    have a better understanding of the air quality in India. The app is designed to show all those facts and relations
    in a very intuitive way.

''')

col1,col2=st.columns(2)
with col1:
    st.markdown("## Historical Analysis")
    st.markdown('''
            The app provides live air quality stats of all the states in India. The stats are updated every hour.
            It also provide the control to the user to apply filters on the data and compare the stats in different ways''')
    
with col2:
    st.markdown("## Forecasting")
    st.markdown(''' .... ''')

st.text('');
st.text('');
st.markdown('# Data Sources')

st.markdown('''
1. **Air Quality Data** - The air quality data is taken from [data.gov.in](https://data.gov.in/). The data is collected using their
            live API and is updated every hour.
2. **Historical Data** - The historical data is taken from [data.gov.in](https://data.gov.in/). The data is collected in various
            files and then compiled into a single dataset for analysis.
''')

