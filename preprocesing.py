import pandas as pd
import numpy as np
import warnings
import itertools
import dateutil
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def load_data(hist_data):

    data=pd.read_csv('historical_data.csv',encoding='unicode_escape')

    nullvalues=data.isnull().sum().sort_values(ascending=False)

    nullvalues_parcentage=(data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)

    missing_value_with_parcent=pd.concat([nullvalues,nullvalues_parcentage],axis=1,keys=['total','parcent'])

    data.drop(['agency'],axis=1,inplace=True)
    data.drop(['stn_code'],axis=1,inplace=True)
    data.drop(['location_monitoring_station'],axis=1,inplace=True)


    data['location']=data['location'].fillna(data['location'].mode()[0])
    data['type']=data['type'].fillna(data['type'].mode()[0])
    # null value imputation for categorical data

    data.fillna(0,inplace=True)
    # null value replaced with zero for the numerical data
    
    return data

def All_affected_area(hist_data):

    data=load_data(hist_data)

    def calculate_si(so2):
        si=0
        if (so2<=40):
            si= so2*(50/40)
        if (so2>40 and so2<=80):
            si= 50+(so2-40)*(50/40)
        if (so2>80 and so2<=380):
            si= 100+(so2-80)*(100/300)
        if (so2>380 and so2<=800):
            si= 200+(so2-380)*(100/800)
        if (so2>800 and so2<=1600):
            si= 300+(so2-800)*(100/800)
        if (so2>1600):
            si= 400+(so2-1600)*(100/800)
        return si
    data['soi']=data['so2'].apply(calculate_si)
    data_01=data[['so2','soi']]
    # data_01.head()
    # calculate individual pollutant index for so2

    def cal_noi(no2):
        ni=0
        if(no2<=40):
            ni= no2*50/40
        elif(no2>40 and no2<=80):
            ni= 50+(no2-14)*(50/40)
        elif(no2>80 and no2<=180):
            ni= 100+(no2-80)*(100/100)
        elif(no2>180 and no2<=280):
            ni= 200+(no2-180)*(100/100)
        elif(no2>280 and no2<=400):
            ni= 300+(no2-280)*(100/120)
        else:
            ni= 400+(no2-400)*(100/120)
        return ni
    data['noi']=data['no2'].apply(cal_noi)
    data_01=data[['no2','noi']]
    # data_01.head()
    # calculate individual pollutant index for no2

    def cal_rspm(rspm):
        rpi=0
        if(rpi<=30):
            rpi=rpi*50/30
        elif(rpi>30 and rpi<=60):
            rpi=50+(rpi-30)*50/30
        elif(rpi>60 and rpi<=90):
            rpi=100+(rpi-60)*100/30
        elif(rpi>90 and rpi<=120):
            rpi=200+(rpi-90)*100/30
        elif(rpi>120 and rpi<=250):
            rpi=300+(rpi-120)*(100/130)
        else:
            rpi=400+(rpi-250)*(100/130)
        return rpi

    data['rpi']=data['rspm'].apply(cal_rspm)
    data_01=data[['rspm','rpi']]
    # data_01.head()

    def cal_spi(spm):
        spi=0
        if(spm<=50):
            spi=spm
        if(spm<50 and spm<=100):
            spi=spm
        elif(spm>100 and spm<=250):
            spi= 100+(spm-100)*(100/150)
        elif(spm>250 and spm<=350):
            spi=200+(spm-250)
        elif(spm>350 and spm<=450):
            spi=300+(spm-350)*(100/80)
        else:
            spi=400+(spm-430)*(100/80)
        return spi
    data['spi']=data['spm'].apply(cal_rspm)
    data_01=data[['spm','spi']]
    # data_01.head()


    def cal_aqi(si,ni,rspmi,spmi):
        aqi=0
        if(si>ni and si>rspmi and si>spmi):
            aqi=si
        if(ni>si and ni>rspmi and ni>spmi):
            aqi=ni
        if rspmi>si and rspmi>ni and rspmi>spmi:
            aqi=rspmi
        if spmi>si and spmi>ni and spmi>rspmi:
            aqi=spmi
        return aqi

    data['AQI']=data.apply(lambda x:cal_aqi(x['soi'],x['noi'],x['rpi'],x['spi']),axis=1)
    data_01=data[['state','soi','noi','rpi','spi','AQI']]

    state=pd.read_csv('poptable.csv')

    dff=pd.merge(state.set_index("state"),data_01.set_index("state"),right_index=True,left_index=True).reset_index()
    
    return dff


def AQI_data(hist_data):

    data=load_data(hist_data)

    def calculate_si(so2):
        si=0
        if (so2<=40):
            si= so2*(50/40)
        if (so2>40 and so2<=80):
            si= 50+(so2-40)*(50/40)
        if (so2>80 and so2<=380):
            si= 100+(so2-80)*(100/300)
        if (so2>380 and so2<=800):
            si= 200+(so2-380)*(100/800)
        if (so2>800 and so2<=1600):
            si= 300+(so2-800)*(100/800)
        if (so2>1600):
            si= 400+(so2-1600)*(100/800)
        return si
    data['soi']=data['so2'].apply(calculate_si)
    data_01=data[['so2','soi']]
    # data_01.head()
    # calculate individual pollutant index for so2

    def cal_noi(no2):
        ni=0
        if(no2<=40):
            ni= no2*50/40
        elif(no2>40 and no2<=80):
            ni= 50+(no2-14)*(50/40)
        elif(no2>80 and no2<=180):
            ni= 100+(no2-80)*(100/100)
        elif(no2>180 and no2<=280):
            ni= 200+(no2-180)*(100/100)
        elif(no2>280 and no2<=400):
            ni= 300+(no2-280)*(100/120)
        else:
            ni= 400+(no2-400)*(100/120)
        return ni
    data['noi']=data['no2'].apply(cal_noi)
    data_01=data[['no2','noi']]
    # data_01.head()
    # calculate individual pollutant index for no2

    def cal_rspm(rspm):
        rpi=0
        if(rpi<=30):
            rpi=rpi*50/30
        elif(rpi>30 and rpi<=60):
            rpi=50+(rpi-30)*50/30
        elif(rpi>60 and rpi<=90):
            rpi=100+(rpi-60)*100/30
        elif(rpi>90 and rpi<=120):
            rpi=200+(rpi-90)*100/30
        elif(rpi>120 and rpi<=250):
            rpi=300+(rpi-120)*(100/130)
        else:
            rpi=400+(rpi-250)*(100/130)
        return rpi

    data['rpi']=data['rspm'].apply(cal_rspm)
    data_01=data[['rspm','rpi']]
    # data_01.head()

    def cal_spi(spm):
        spi=0
        if(spm<=50):
            spi=spm
        if(spm<50 and spm<=100):
            spi=spm
        elif(spm>100 and spm<=250):
            spi= 100+(spm-100)*(100/150)
        elif(spm>250 and spm<=350):
            spi=200+(spm-250)
        elif(spm>350 and spm<=450):
            spi=300+(spm-350)*(100/80)
        else:
            spi=400+(spm-430)*(100/80)
        return spi
    data['spi']=data['spm'].apply(cal_rspm)
    data_01=data[['spm','spi']]
    # data_01.head()


    def cal_aqi(si,ni,rspmi,spmi):
        aqi=0
        if(si>ni and si>rspmi and si>spmi):
            aqi=si
        if(ni>si and ni>rspmi and ni>spmi):
            aqi=ni
        if rspmi>si and rspmi>ni and rspmi>spmi:
            aqi=rspmi
        if spmi>si and spmi>ni and spmi>rspmi:
            aqi=spmi
        return aqi

    data['AQI']=data.apply(lambda x:cal_aqi(x['soi'],x['noi'],x['rpi'],x['spi']),axis=1)
    data_01=data[['state','soi','noi','rpi','spi','AQI']]
    # data_01.head()

    def AQI_range(x):
        if x<=50:
            return "Good"
        elif x>50 and x<=100:
            return "Moderate"
        elif x>100 and x<=200:
            return "Poor"
        elif x>200 and x<=300:
            return "unhealthy"
        elif x>300 and x<=400:
            return "Very Unhealthy"
        elif x>400:
            return "Hazardous"
        
    data['AQI_range']=data['AQI'].apply(AQI_range)
    # data.head()
    state=pd.read_csv('poptable.csv')

    dff=pd.merge(state.set_index('state'),data.set_index('state'),right_index=True,left_index=True).reset_index()

    data=dff

    return data

def mean_year_data(data):

    # data=load_data('historical_data')
    df=data[['AQI','date']]
    df["date"] = pd.to_datetime(df['date'])

    #Calculating the yearly mean for the data 
    df=df.set_index('date').resample('M')["AQI"].mean()
    # df.head()

    #preprocessing the data values
    data=df.reset_index(level=0, inplace=False)
    data = data[np.isfinite(data['AQI'])]
    data=data[data.date != '1970-01-31']
    data = data.reset_index(drop=True)

    # Convert 'date' column to datetime, coercing errors
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['month']=data['date'].dt.month
    data['year']=data['date'].dt.year
    data['year']=data['year'].fillna(0.0).astype(int)

    data=data[(data['year']>0)]


    return data








