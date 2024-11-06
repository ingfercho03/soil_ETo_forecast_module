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
    dataset = 'JAXA/GPM_L3/GSMaP/v7/operational'
    sensor = ['hourlyPrecipRate']
    df_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
    df_time_series['Date_format']= pd.to_datetime(df_time_series['Date'], format='%b %d %Y')
    df_time_series = df_time_series.set_index('Date_format')
    Precipitacion_day = df_time_series['Sensor_Values'].resample('D').sum()
    df_time_Precipitacion_day = Precipitacion_day.to_frame()
    df_precipitacion_day = check_datos(start_date,end_date,df_time_Precipitacion_day)
    #print(df_precipitacion_day)
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

def kelvin_to_celsius(value):
    value = value - 273.15
    return value
def pascales_to_kpa(value):
    pressure_kpa = value / 1000.0
    return pressure_kpa

def df_eto_CFSV2(start_date, end_date,roi,lon,lat):
  dataset = 'NOAA/CFSV2/FOR6H'
  sensor=['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Minimum_temperature_height_above_ground_6_Hour_Interval','Maximum_temperature_height_above_ground_6_Hour_Interval','Pressure_surface']
  dict_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
  list_variables = list(dict_time_series.keys())
  df_var_eto_CFSV2 = pd.DataFrame()
  for i in list_variables:
      dat_dict=dict_time_series.get(i)
      df = pd.DataFrame(dat_dict, columns = ['Date', 'Sensor_Values'])
      df = df.rename(columns={'Sensor_Values': i})
      df = df.rename(columns={'Date': 'Date'+'_'+i})
      df_var_eto_CFSV2= pd.concat([df_var_eto_CFSV2, df],axis=1)
  df_var_eto_CFSV2
  df_var_eto_CFSV2 = df_var_eto_CFSV2.drop(['Date_v-component_of_wind_height_above_ground','Date_Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Date_Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Date_Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Date_Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Date_Minimum_temperature_height_above_ground_6_Hour_Interval','Date_Maximum_temperature_height_above_ground_6_Hour_Interval','Date_Pressure_surface'],axis=1)
  df_var_eto_CFSV2['Date']=df_var_eto_CFSV2['Date_u-component_of_wind_height_above_ground'].dt.date
  df_var_eto_CFSV2 = df_var_eto_CFSV2.drop(['Date_u-component_of_wind_height_above_ground'],axis=1)
  def watts_to_jMoules_6hours(value):
    conversion_factor = 0.0216
    return value * conversion_factor
  df_var_eto_CFSV2['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average']= df_var_eto_CFSV2['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average'].apply(watts_to_jMoules_6hours)
  df_var_eto_CFSV2['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']= df_var_eto_CFSV2['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'].apply(watts_to_jMoules_6hours)
  df_var_eto_CFSV2['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average']= df_var_eto_CFSV2['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'].apply(watts_to_jMoules_6hours)
  df_var_eto_CFSV2['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']= df_var_eto_CFSV2['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'].apply(watts_to_jMoules_6hours)
  df_var_eto_CFSV2['Minimum_temperature_height_above_ground_6_Hour_Interval'] = df_var_eto_CFSV2['Minimum_temperature_height_above_ground_6_Hour_Interval'].apply(kelvin_to_celsius)
  df_var_eto_CFSV2['Maximum_temperature_height_above_ground_6_Hour_Interval'] = df_var_eto_CFSV2['Maximum_temperature_height_above_ground_6_Hour_Interval'].apply(kelvin_to_celsius)
  df_var_eto_CFSV2['Pressure_surface'] = df_var_eto_CFSV2['Pressure_surface'].apply(pascales_to_kpa)
  df_var_eto_CFSV2_by_day = df_var_eto_CFSV2.groupby('Date').agg({
    'u-component_of_wind_height_above_ground': 'max',
    'Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average': 'sum',
    'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average': 'sum',
    'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average': 'sum',
    'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average': 'sum',
    'v-component_of_wind_height_above_ground': 'max',
    'Minimum_temperature_height_above_ground_6_Hour_Interval': 'min',
    'Maximum_temperature_height_above_ground_6_Hour_Interval': 'max',
    'Pressure_surface': 'max' })
  df_var_eto_CFSV2_by_day['net_radiation'] = (df_var_eto_CFSV2_by_day['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'] - df_var_eto_CFSV2_by_day['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']) - (df_var_eto_CFSV2_by_day['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average']- df_var_eto_CFSV2_by_day['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'])
  df_var_eto_CFSV2_by_day['wind_speed']= np.sqrt(df_var_eto_CFSV2_by_day['u-component_of_wind_height_above_ground']**2 + df_var_eto_CFSV2_by_day['v-component_of_wind_height_above_ground']**2) 
  df_var_eto_CFSV2_by_day= df_var_eto_CFSV2_by_day.rename(columns={'Minimum_temperature_height_above_ground_6_Hour_Interval':'T_min','Maximum_temperature_height_above_ground_6_Hour_Interval':'T_max','wind_speed':'U_z','net_radiation':'R_n'})#'Pressure_surface':'P'
  freq='D'
  z_u=10
  et1 = ETo()
  df_var_eto_CFSV2_by_day.index=pd.to_datetime(df_var_eto_CFSV2_by_day.index)
  et1.param_est(df_var_eto_CFSV2_by_day, freq, lat, lon, z_u)
  eto1 = et1.eto_fao()
  df_eto_CFSV2=eto1.to_frame()
  df_eto_CFSV2= df_eto_CFSV2.rename(columns={'ETo_FAO_mm':'Sensor_Values'})
  return df_eto_CFSV2

def eto_var(start_date, end_date,roi,lon,lat):
    dataset = 'ECMWF/ERA5_LAND/DAILY_AGGR'
    sensor = ['surface_net_solar_radiation_sum','surface_net_thermal_radiation_sum','temperature_2m_min','temperature_2m_max','u_component_of_wind_10m_max','v_component_of_wind_10m_max','dewpoint_temperature_2m','surface_pressure']
    dict_time_series = series_tiempo(start_date,end_date,roi,dataset,sensor)
    list_variables = list(dict_time_series.keys())
    df_var_eto = pd.DataFrame()
    for i in list_variables:
        dat_dict=dict_time_series.get(i)
        df = pd.DataFrame(dat_dict, columns = ['Date', 'Sensor_Values'])
        df = df.rename(columns={'Sensor_Values': i})
        df = df.rename(columns={'Date': 'Date'+'_'+i})
        df_var_eto= pd.concat([df_var_eto, df],axis=1)
    def joules_to_megajoules(energy_joules):
        energy_megajoules = energy_joules / 1e6
        return energy_megajoules
    df_var_eto = df_var_eto.drop(['Date_temperature_2m_min','Date_surface_net_thermal_radiation_sum','Date_v_component_of_wind_10m_max','Date_temperature_2m_max','Date_u_component_of_wind_10m_max','Date_dewpoint_temperature_2m','Date_surface_pressure'],axis=1)
    df_var_eto['surface_net_solar_radiation_sum']= df_var_eto['surface_net_solar_radiation_sum'].apply(joules_to_megajoules)
    df_var_eto['surface_net_thermal_radiation_sum']= df_var_eto['surface_net_thermal_radiation_sum'].apply(joules_to_megajoules)
    df_var_eto['surface_net_radiation_sum']= df_var_eto['surface_net_solar_radiation_sum'] - df_var_eto['surface_net_thermal_radiation_sum']
    df_var_eto['wind_speed'] = np.sqrt(df_var_eto['u_component_of_wind_10m_max']**2 + df_var_eto['v_component_of_wind_10m_max']**2)
    df_var_eto['temperature_2m_min'] = df_var_eto['temperature_2m_min'].apply(kelvin_to_celsius)
    df_var_eto['temperature_2m_max'] = df_var_eto['temperature_2m_max'].apply(kelvin_to_celsius)
    df_var_eto['dewpoint_temperature_2m'] = df_var_eto['dewpoint_temperature_2m'].apply(kelvin_to_celsius)
    df_var_eto['surface_pressure'] = df_var_eto['surface_pressure'].apply(pascales_to_kpa)
    df_var_eto= df_var_eto.rename(columns={'Date_surface_net_solar_radiation_sum':'Date','surface_net_radiation_sum':'R_n','temperature_2m_min':'T_min','temperature_2m_max':'T_max','dewpoint_temperature_2m':'T_dew','wind_speed':'U_z'})#'surface_pressure':'P'

    freq='D'
    z_u=10
    et1 = ETo()
    df_var_eto=df_var_eto.set_index('Date')
    et1.param_est(df_var_eto, freq, lat, lon, z_u)
    eto1 = et1.eto_fao()
    df_eto=eto1.to_frame()
    df_eto= df_eto.rename(columns={'ETo_FAO_mm':'Sensor_Values'})
    start_date_new_data = (df_eto.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    df_eto_var_CFSV2 = df_eto_CFSV2(start_date_new_data, end_date,roi,lon,lat)
    #print(df_eto_CFSV2)
    df_eto_concat= pd.concat([df_eto, df_eto_var_CFSV2],axis=0)
    #df_eto_his = check_datos(start_date,end_date,df_eto)
    df_eto_his = check_datos(start_date,end_date,df_eto_concat)
    #return df_eto_his
    return df_eto_his

@app.get("/")

def index():
    return{"message" : "Completar la URL y campos"}

@app.get("/satweather/")

def get_sat_data(lat: float,lon: float):

    service_account = 'xxxxx'
    credentials = ee.ServiceAccountCredentials(service_account, 'xxxxx')
    ee.Initialize(credentials)

    # Region of interest (ROI)
    #roi = ee.Geometry.Point(-74.9840391, 5.1978685)  # Finca San Luis
    roi = ee.Geometry.Point(lon, lat) 
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    seven_days_ago = current_date - timedelta(days=10)
    start_date= seven_days_ago.strftime("%Y-%m-%d")
    precip_his = precipitaciones(start_date, end_date,roi)
    humedad_suelo_his = humedad_suelo(start_date, end_date,roi)
    df_eto_his = df_eto_CFSV2(start_date, end_date,roi,lon,lat)
    eto_his = check_datos(start_date,end_date,df_eto_his)
    #eto_his = eto_var(start_date, end_date,roi,lon,lat)
    print(eto_his)

    dict_his={}
    for i, j in enumerate (precip_his.index):
        ts = str(j)
        precipitacion= precip_his['Sensor_Values'].loc[ts].round(3)
        humedadSuelo= humedad_suelo_his['Sensor_Values'].loc[ts].round(3)
        etoVar = eto_his['Sensor_Values'].loc[ts].round(3)
        print(etoVar)
        if np.isnan(precipitacion):
            precipitacion = None
        if np.isnan(humedadSuelo):
           humedadSuelo = None 
        if np.isnan(etoVar):
            print(etoVar)
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

def model_predict_SM(roi):
    model_predict_SM = pickle.load(open('modelo_humedad_suelo.pkl','rb'))
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=18)
    start_date= days_ago.strftime("%Y-%m-%d")
    soilM_his = humedad_suelo(start_date, end_date,roi)
    soilM_his.dropna(inplace=True)
    soilM_his.rename_axis("Date",inplace=True)
    soilM_his=soilM_his.asfreq('d')
    last_window = soilM_his['Sensor_Values'].tail(11)
    steps=5
    prediction= model_predict_SM.predict(steps,last_window=last_window)
    return prediction

def model_predict_ETo(roi,lon,lat):
    model_predict_ETo = pickle.load(open('modelo_ETo-2.pkl','rb'))
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=31)
    start_date= days_ago.strftime("%Y-%m-%d")
    ETo_his = eto_var(start_date, end_date,roi,lon,lat)
    ETo_his.dropna(inplace=True)
    ETo_his.rename_axis("Date",inplace=True)
    ETo_his=ETo_his.asfreq('d')
    last_window = ETo_his['Sensor_Values'].tail(30)
    steps=4
    prediction= model_predict_ETo.predict(steps,last_window=last_window)
    return prediction

@app.get("/forecastWeather/")
def modelos_predict(lat:float, lon:float):
    service_account = 'xxxxx'
    credentials = ee.ServiceAccountCredentials(service_account, 'xxxxx')
    ee.Initialize(credentials)
    roi = ee.Geometry.Point(lon, lat)
    #precipi_predict= model_predict_precipitacion(roi)
    soilM_predict=model_predict_SM(roi)
    ETo_predict=model_predict_ETo(roi,lon,lat)
    print(soilM_predict)
    print(ETo_predict)
    dict_predict={}
    current_day=datetime.now().date().strftime("%Y-%m-%d")
    i=0
    for j in (ETo_predict.index):
        ts= str(j)
        if ts >= current_day:
            ETo_predict_value= ETo_predict[ts]
            SM_predict_value= soilM_predict[ts]
            dict_predict["Dato{}".format(i)]={"ts":ts,"ETo_pred":ETo_predict_value,"SM_pred":SM_predict_value}
            i+=1
    return dict_predict
    
    


