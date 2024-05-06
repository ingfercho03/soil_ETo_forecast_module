from fastapi import FastAPI
import ee
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from eto import ETo
import pickle
import json
from pydantic import BaseModel

app = FastAPI()

def series_tiempo(start_date, end_date, roi, dataset, sensor):
  
    collection = (ee.ImageCollection(dataset)
              .filterBounds(roi)
              .filterDate(ee.Date(start_date), ee.Date(end_date))
              )
    sorted_collection = collection.sort('system:time_start')
    time_start = datetime.now()
    dict_values={}
    for i in sensor:
        dates = []
        sensor_values = []
        for image in sorted_collection.toList(sorted_collection.size()).getInfo():
            ee_image = ee.Image(image['id'])
            date_str = ee_image.get('system:time_start').getInfo()
            # Parse the string date into a Python datetime object
            acquisition_date = datetime.utcfromtimestamp(int(date_str) // 1000)
            dates.append(acquisition_date)
            sensor_band = ee_image.select(i)\
                              .reduceRegion(ee.Reducer.first(), roi,1)\
                              .get(i)
            sensor_values.append(sensor_band.getInfo())
        dict_values[i] = {'Date': dates, 'Sensor_Values': sensor_values}
    if (len(dict_values)!= 1):
        return dict_values
    else:
        dat_dict_1=dict_values.get(sensor[0])
        df = pd.DataFrame(dat_dict_1, columns = ['Date', 'Sensor_Values'])
        return df


def check_datos(start_date,end_date,df):
    list_dates_values=df.index.tolist()
    list_dates_values_str=[]
    for i in range(0,len(list_dates_values)):
        list_dates_values_str.append(list_dates_values[i].strftime('%Y-%m-%d'))
    list_dates_values_str
    date_list = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    while start <= end:
        date_list.append(start.strftime('%Y-%m-%d'))
        start += timedelta(days=1)
    set1 = set(list_dates_values_str)
    set2 = set(date_list)
    differences = set1.symmetric_difference(set2)
    if (len(differences)!=0):
        new_rows_data =[]
        new_indexes=pd.to_datetime(list(differences))
        for i in range(len(differences)):
            new_rows_data.append({'Sensor_Values': None})
        new_rows_df = pd.DataFrame(new_rows_data, index=new_indexes)
        df_sensor = pd.concat([df, new_rows_df]).sort_index()
    else:
        df_sensor=df
    return df_sensor

def precipitaciones(start_date, end_date,roi):
    dataset = 'JAXA/GPM_L3/GSMaP/v6/operational'
    sensor = ['hourlyPrecipRate']
    df_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
    df_time_series['Date_format']= pd.to_datetime(df_time_series['Date'], format='%b %d %Y')
    df_time_series = df_time_series.set_index('Date_format')
    Precipitacion_day = df_time_series['Sensor_Values'].resample('D').sum()
    df_time_Precipitacion_day = Precipitacion_day.to_frame()
    df_precipitacion_day = check_datos(start_date,end_date,df_time_Precipitacion_day)
    return df_precipitacion_day
    #return df_time_Precipitacion_day['Sensor_Values'].to_dict()

def humedad_suelo(start_date, end_date,roi):
    dataset = 'NASA/SMAP/SPL4SMGP/007'
    sensor = ['sm_rootzone']
    df_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
    df_time_series['Date_day']=df_time_series['Date'].dt.date
    df_time_series_humedad_mean_day= df_time_series.groupby('Date_day')['Sensor_Values'].mean().reset_index()
    df_time_series_humedad_mean_day['Date_day']= pd.to_datetime(df_time_series_humedad_mean_day['Date_day'])
    df_time_series_humedad_mean_day=df_time_series_humedad_mean_day.set_index('Date_day')
    df_humedad_suelo = check_datos(start_date,end_date,df_time_series_humedad_mean_day)
    return df_humedad_suelo

def eto_var(start_date, end_date,roi,lon,lat):
    dataset = 'ECMWF/ERA5_LAND/DAILY_AGGR'
    sensor = ['surface_net_solar_radiation_sum','temperature_2m_min','temperature_2m_max','u_component_of_wind_10m_max','dewpoint_temperature_2m','surface_pressure']
    dict_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
    list_variables = list(dict_time_series.keys())
    df_var_eto = pd.DataFrame()
    for i in list_variables:
        dat_dict=dict_time_series.get(i)
        df = pd.DataFrame(dat_dict, columns = ['Date', 'Sensor_Values'])
        df = df.rename(columns={'Sensor_Values': i})
        df = df.rename(columns={'Date': 'Date'+'_'+i})
        df_var_eto= pd.concat([df_var_eto, df],axis=1)
    df_var_eto = df_var_eto.drop(['Date_temperature_2m_min','Date_temperature_2m_max','Date_u_component_of_wind_10m_max','Date_dewpoint_temperature_2m','Date_surface_pressure'],axis=1)
    def joules_to_megajoules(energy_joules):
        energy_megajoules = energy_joules / 1e6
        return energy_megajoules
    def kelvin_to_celsius(value):
        value = value - 273.15
        return value
    def pascales_to_kpa(value):
        pressure_kpa = value / 1000.0
        return pressure_kpa
    df_var_eto['surface_net_solar_radiation_sum']= df_var_eto['surface_net_solar_radiation_sum'].apply(joules_to_megajoules)
    df_var_eto['temperature_2m_min'] = df_var_eto['temperature_2m_min'].apply(kelvin_to_celsius)
    df_var_eto['temperature_2m_max'] = df_var_eto['temperature_2m_max'].apply(kelvin_to_celsius)
    df_var_eto['dewpoint_temperature_2m'] = df_var_eto['dewpoint_temperature_2m'].apply(kelvin_to_celsius)
    df_var_eto['surface_pressure'] = df_var_eto['surface_pressure'].apply(pascales_to_kpa)
    df_var_eto= df_var_eto.rename(columns={'Date_surface_net_solar_radiation_sum':'Date','surface_net_solar_radiation_sum':'R_n','temperature_2m_min':'T_min','temperature_2m_max':'T_max','dewpoint_temperature_2m':'T_dew','u_component_of_wind_10m_max':'U_z','surface_pressure':'P'})

    freq='D'
    z_u=10
    et1 = ETo()
    df_var_eto=df_var_eto.set_index('Date')
    et1.param_est(df_var_eto, freq, lat, lon, z_u)
    eto1 = et1.eto_fao()
    df_eto=eto1.to_frame()
    df_eto= df_eto.rename(columns={'ETo_FAO_mm':'Sensor_Values'})
    df_eto_his = check_datos(start_date,end_date,df_eto)
    return df_eto_his


@app.get("/")

def index():
    return{"message" : "Completar la URL y campos"}

@app.get("/satweather/")

def get_sat_data(lat: float,lon: float):

    service_account = 'ingferchogee@ee-ingfercho03.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'ee-ingfercho03-823327c28642.json')
    ee.Initialize(credentials)

    # Region of interest (ROI)
    #roi = ee.Geometry.Point(-74.9840391, 5.1978685)  # Finca San Luis
    roi = ee.Geometry.Point(lon, lat) 
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    seven_days_ago = current_date - timedelta(days=15)
    start_date= seven_days_ago.strftime("%Y-%m-%d")
    precip_his = precipitaciones(start_date, end_date,roi)
    humedad_suelo_his = humedad_suelo(start_date, end_date,roi)
    eto_his = eto_var(start_date, end_date,roi,lon,lat)

    dict_his={}
    for i, j in enumerate (precip_his.index):
        ts = str(j)
        precipitacion= precip_his['Sensor_Values'].loc[ts].round(3)
        humedadSuelo= humedad_suelo_his['Sensor_Values'].loc[ts].round(3)
        etoVar = eto_his['Sensor_Values'].loc[ts].round(3)
        if np.isnan(precipitacion):
            precipitacion = None
        if np.isnan(humedadSuelo):
           humedadSuelo = None 
        if np.isnan(etoVar):
            etoVar= None
        dict_his["Dato{}".format(i)]= {"ts":ts, "Precipitacion":precipitacion, "Humedad_Suelo":humedadSuelo, "ETo":etoVar}
    
    return dict_his

def model_predict_precipitacion(roi):
    model_predict_Precip = pickle.load(open('modelo_Precipitaciones.pkl','rb'))
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=7)
    start_date= days_ago.strftime("%Y-%m-%d")
    precip_his = precipitaciones(start_date, end_date,roi)
    precip_his.dropna(inplace=True)
    precip_his.rename_axis("Date",inplace=True)
    precip_his=precip_his.asfreq('d')
    last_window = precip_his['Sensor_Values'].tail(3)
    print(last_window)
    print(type(last_window))
    steps=3
    prediction= model_predict_Precip.predict(steps,last_window=last_window)
    return prediction


#model_predict_ETo = pickle.load(open('modelo_ETo.pkl','rb'))
#model_predict_HS = pickle.load(open('modelo_humedad_suelo.pkl','rb'))


@app.get("/forecastWeather/")
def modelos_predict(lat:float, lon:float):
    service_account = 'ingferchogee@ee-ingfercho03.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'ee-ingfercho03-823327c28642.json')
    ee.Initialize(credentials)
    roi = ee.Geometry.Point(lon, lat)
    precipi_predict= model_predict_precipitacion(roi)
    return precipi_predict
    
    


