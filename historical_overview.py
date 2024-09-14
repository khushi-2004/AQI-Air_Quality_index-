import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import dateutil
import statsmodels.api as sm
import matplotlib.dates as mdates

import plotly.graph_objects as go

import streamlit as st
import base64
import preprocesing
from prophet import Prophet
from mpl_toolkits.basemap import Basemap

st.set_page_config(layout="wide",
                   page_icon="",page_title="Historical Analysis")


@st.cache_data
def get_image_as_base64(file):
    with open(file,'rb') as f:
        data=f.read()
    return base64.b64encode(data).decode()

# img=get_image_as_base64('')
st.text('');
st.text('');

# st.markdown("""<h1> </h1> """)

df=preprocesing.load_data('historical_data.csv')
# st.dataframe(df)

st.text('');
st.text('');

dff=preprocesing.All_affected_area('historical_data.csv')

# Create a title in Streamlit
st.title("Map of Affected Areas")

# Initialize the Basemap
m = Basemap(projection='mill', llcrnrlat=5, urcrnrlat=40, llcrnrlon=60, urcrnrlon=110, lat_ts=20, resolution='c')

# Sample data (Replace `dff` with actual DataFrame in your app)
longitudes = dff["latitude"].tolist()  # Replace with your data
latitudes = dff["longitude"].tolist()  # Replace with your data

# Convert latitude and longitude to map projection coordinates
x, y = m(longitudes, latitudes)

# Create the plot
fig = plt.figure(figsize=(12, 10))
plt.title("All affected areas")

# Plot the data points
m.plot(x, y, "o", markersize=3, color='blue')

# Draw coastlines, continents, and boundaries
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()

# Get the map corners in map projection coordinates
map_width = m.xmax - m.xmin
map_height = m.ymax - m.ymin

# Define positions for the compass directions relative to the map boundaries
compass_positions = {
    'N': (m.xmin + map_width / 2, m.ymax * 0.98),
    'S': (m.xmin + map_width / 2, m.ymin + map_height * 0.02),
    'E': (m.xmax * 0.98, m.ymin + map_height / 2),
    'W': (m.xmin + map_width * 0.02, m.ymin + map_height / 2),
}

# Add text annotations for the compass directions
for direction, position in compass_positions.items():
    plt.text(position[0], position[1], direction, ha='center', va='center', fontsize=12, fontweight='bold')

# Display the plot in Streamlit
st.plotly_chart(fig,use_container_width=True)


st.text('');
st.text('');
st.text('');
st.text('');


data=preprocesing.AQI_data('historical_data')
# st.dataframe(data)

#Visualization of AQI across india 

st.title("Visualization of AQI across india")
data['date'] = pd.to_datetime(data['date'],errors='coerce') # date parse

data['year'] = data['date'].dt.year # year
data['year'] = data['year'].fillna(0.0).astype(int)
data = data[(data['year']>0)]

dff = data[['AQI','year']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
# Create a Plotly figure
fig = go.Figure()

# Add a scatter trace to the figure
fig.add_trace(go.Scatter(x=dff["year"], y=dff["AQI"], mode='lines+markers', name='AQI'))

# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.title("SO2 level in every state")
col1,col2=st.columns(2)
with col1:
    st.header("so2 level")
    fig,ax=plt.subplots()
    ax=sns.barplot(x='state',y='so2',data=df)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)
with col2:
    st.header("no2 level")
    fig,ax=plt.subplots()
    ax=sns.barplot(x='state',y='no2',data=df)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# st.dataframe(data)

st.title("yearly mean of AQI")
year_data=preprocesing.mean_year_data(data)
mean_year_df=data[['AQI','year']].groupby(['year']).median().reset_index().sort_values(by='year',ascending=False)

# st.dataframe(mean_year_df)

fig,ax=plt.subplots()
ax=plt.plot(mean_year_df['year'],mean_year_df['AQI'])
plt.xticks(rotation=80)
st.pyplot(fig)

# st.title("Forcasting the data")

# # st.dataframe(year_data.tail())
# forcast_data=data[['date','AQI']]
# forcast_data
# forcast_data.tail()
# m=Prophet()
# forcast_data.columns=['ds','y']

# model=m.fit(forcast_data)

# future=m.make_future_dataframe(periods=1500,freq='D')
# forecast=m.predict(future)
# forecast.tail()

# from prophet.plot import plot_plotly
# # import streamlit as st

# # Assuming 'm' is your trained Prophet model and 'forecast' is the dataframe with forecasted data.
# fig = plot_plotly(m, forecast)  # Create the interactive Plotly figure

# st.plotly_chart(fig)  # Use Streamlit to display the Plotly figure

# import streamlit as st
# import pandas as pd
# from prophet import Prophet
from prophet.plot import plot_plotly

# Title for the Streamlit app
st.title("Forecasting the Data")

# Load and prepare data (for example, assuming `data` is already defined)
forcast_data = data[['date', 'AQI']]

# Display the last few rows of the data (optional)
# st.write(forcast_data.tail())

# Prepare the data for Prophet
forcast_data.columns = ['ds', 'y']  # Prophet expects columns 'ds' and 'y'
m = Prophet()  # Initialize Prophet model

# Fit the model to your data
model = m.fit(forcast_data)

# Make future dataframe and forecast
future = m.make_future_dataframe(periods=500, freq='D')  # Extend the dataframe for future predictions
forecast = m.predict(future)

# Display the last few rows of the forecast
# st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# st.write(forcast_data.head())  # Check if data is loaded correctly
st.write(forecast.tail())  # Ensure that the forecast is generated correctly

# Plot the forecast with Plotly and display in Streamlit
# fig = plot_plotly(m, forecast)  # Create the interactive Plotly figure
# st.plotly_chart(fig)  # Display the figure in Streamlit


