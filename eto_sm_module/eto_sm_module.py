"""This module retrieves ET₀ and SM from satellite products and forecasts them using ML models"""
from datetime import datetime, timedelta
import pickle
import numpy as np
from eto import ETo
import pandas as pd
import ee
from fastapi import FastAPI
from eto_sm_module import models
#from importlib.resources import files
import importlib.resources as pkg_resources
#import json
#import os
#from pydantic import BaseModel


def series_time(start_date, end_date, roi, dataset, sensor):
    """This function collects data from satellite products and present in time series"""
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
    if len(dict_values)!= 1:
        return dict_values
    else:
        dat_dict_1=dict_values.get(sensor[0])
        df = pd.DataFrame(dat_dict_1, columns = ['Date', 'Sensor_Values'])
        return df


def check_data(start_date,end_date,df):
    """This function review nulls or missing data"""
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
    if len(differences)!=0:
        new_rows_data =[]
        new_indexes=pd.to_datetime(list(differences))
        for i in range(len(differences)):
            new_rows_data.append({'Sensor_Values': None})
        new_rows_df = pd.DataFrame(new_rows_data, index=new_indexes)
        df_sensor = pd.concat([df, new_rows_df]).sort_index()
    else:
        df_sensor=df
    return df_sensor

def precipitation(start_date, end_date,roi):
    """This function get rain data"""
    dataset = 'JAXA/GPM_L3/GSMaP/v7/operational'
    sensor = ['hourlyPrecipRate']
    df_time_series = series_time(start_date,end_date,roi,dataset,sensor)
    df_time_series['Date_format']= pd.to_datetime(df_time_series['Date'], format='%b %d %Y')
    df_time_series = df_time_series.set_index('Date_format')
    precipitacion_day = df_time_series['Sensor_Values'].resample('D').sum()
    df_time_precipitacion_day = precipitacion_day.to_frame()
    df_precipitacion_day = check_data(start_date,end_date,df_time_precipitacion_day)
    #print(df_precipitacion_day)
    return df_precipitacion_day
    #return df_time_precipitacion_day['Sensor_Values'].to_dict()

def soil_moisturefun(start_date, end_date,roi):
    """This function get soil moisture"""
    dataset = 'NASA/SMAP/SPL4SMGP/007'
    sensor = ['sm_rootzone']
    df_time_series = series_time(start_date,end_date,roi,dataset,sensor)
    df_time_series['Date_day']=df_time_series['Date'].dt.date
    df_tseries_sm_mean_day= df_time_series.groupby('Date_day')['Sensor_Values'].mean().reset_index()
    df_tseries_sm_mean_day['Date_day']= pd.to_datetime(df_tseries_sm_mean_day['Date_day'])
    df_tseries_sm_mean_day=df_tseries_sm_mean_day.set_index('Date_day')
    df_humedad_suelo = check_data(start_date,end_date,df_tseries_sm_mean_day)
    return df_humedad_suelo

def kelvin_to_celsius(value):
    """This function transforms from kelvin to celsius"""
    value = value - 273.15
    return value
def pascales_to_kpa(value):
    """This function transforms from pascales to kpa"""
    pressure_kpa = value / 1000.0
    return pressure_kpa

def df_eto_cfsv2(start_date, end_date,roi,lon,lat):
    """This function gets climate data"""
    dataset = 'NOAA/CFSV2/FOR6H'
    sensor=['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','u-component_of_wind_height_above_ground','v-component_of_wind_height_above_ground','Minimum_temperature_height_above_ground_6_Hour_Interval','Maximum_temperature_height_above_ground_6_Hour_Interval','Pressure_surface']
    dict_time_series = series_time(start_date,end_date,roi,dataset,sensor)
    list_variables = list(dict_time_series.keys())
    df_var_eto_cfsv2 = pd.DataFrame()
    for i in list_variables:
        dat_dict=dict_time_series.get(i)
        df = pd.DataFrame(dat_dict, columns = ['Date', 'Sensor_Values'])
        df = df.rename(columns={'Sensor_Values': i})
        df = df.rename(columns={'Date': 'Date'+'_'+i})
        df_var_eto_cfsv2= pd.concat([df_var_eto_cfsv2, df],axis=1)
    df_var_eto_cfsv2
    df_var_eto_cfsv2 = df_var_eto_cfsv2.drop(['Date_v-component_of_wind_height_above_ground','Date_Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Date_Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Date_Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average','Date_Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Date_Minimum_temperature_height_above_ground_6_Hour_Interval','Date_Maximum_temperature_height_above_ground_6_Hour_Interval','Date_Pressure_surface'],axis=1)
    df_var_eto_cfsv2['Date']=df_var_eto_cfsv2['Date_u-component_of_wind_height_above_ground'].dt.date
    df_var_eto_cfsv2 = df_var_eto_cfsv2.drop(['Date_u-component_of_wind_height_above_ground'],axis=1)
    def watts_to_joules_6hours(value):
        conversion_factor = 0.0216
        return value * conversion_factor
    df_var_eto_cfsv2['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average']= df_var_eto_cfsv2['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average'].apply(watts_to_joules_6hours)
    df_var_eto_cfsv2['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']= df_var_eto_cfsv2['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'].apply(watts_to_joules_6hours)
    df_var_eto_cfsv2['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average']= df_var_eto_cfsv2['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'].apply(watts_to_joules_6hours)
    df_var_eto_cfsv2['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']= df_var_eto_cfsv2['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'].apply(watts_to_joules_6hours)
    df_var_eto_cfsv2['Minimum_temperature_height_above_ground_6_Hour_Interval'] = df_var_eto_cfsv2['Minimum_temperature_height_above_ground_6_Hour_Interval'].apply(kelvin_to_celsius)
    df_var_eto_cfsv2['Maximum_temperature_height_above_ground_6_Hour_Interval'] = df_var_eto_cfsv2['Maximum_temperature_height_above_ground_6_Hour_Interval'].apply(kelvin_to_celsius)
    df_var_eto_cfsv2['Pressure_surface'] = df_var_eto_cfsv2['Pressure_surface'].apply(pascales_to_kpa)
    df_var_eto_cfsv2_by_day = df_var_eto_cfsv2.groupby('Date').agg({
      'u-component_of_wind_height_above_ground': 'max',
      'Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average': 'sum',
      'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average': 'sum',
      'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average': 'sum',
      'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average': 'sum',
      'v-component_of_wind_height_above_ground': 'max',
      'Minimum_temperature_height_above_ground_6_Hour_Interval': 'min',
      'Maximum_temperature_height_above_ground_6_Hour_Interval': 'max',
      'Pressure_surface': 'max' })
    df_var_eto_cfsv2_by_day['net_radiation'] = (df_var_eto_cfsv2_by_day['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'] - df_var_eto_cfsv2_by_day['Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average']) - (df_var_eto_cfsv2_by_day['Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average']- df_var_eto_cfsv2_by_day['Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'])
    df_var_eto_cfsv2_by_day['wind_speed']= np.sqrt(df_var_eto_cfsv2_by_day['u-component_of_wind_height_above_ground']**2 + df_var_eto_cfsv2_by_day['v-component_of_wind_height_above_ground']**2)
    df_var_eto_cfsv2_by_day= df_var_eto_cfsv2_by_day.rename(columns={'Minimum_temperature_height_above_ground_6_Hour_Interval':'T_min','Maximum_temperature_height_above_ground_6_Hour_Interval':'T_max','wind_speed':'U_z','net_radiation':'R_n'})
    freq='D'
    z_u=10
    et1 = ETo()
    df_var_eto_cfsv2_by_day.index=pd.to_datetime(df_var_eto_cfsv2_by_day.index)
    et1.param_est(df_var_eto_cfsv2_by_day, freq, lat, lon, z_u)
    eto1 = et1.eto_fao()
    df_eto_cfsv2=eto1.to_frame()
    df_eto_cfsv2= df_eto_cfsv2.rename(columns={'ETo_FAO_mm':'Sensor_Values'})
    return df_eto_cfsv2

def eto_var(start_date, end_date,roi,lon,lat):
    """This function get and calculate eto"""
    dataset = 'ECMWF/ERA5_LAND/DAILY_AGGR'
    sensor = ['surface_net_solar_radiation_sum','surface_net_thermal_radiation_sum','temperature_2m_min','temperature_2m_max','u_component_of_wind_10m_max','v_component_of_wind_10m_max','dewpoint_temperature_2m','surface_pressure']
    dict_time_series = series_time(start_date,end_date,roi,dataset,sensor)
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
    df_eto_var_cfsv2 = df_eto_cfsv2(start_date_new_data, end_date,roi,lon,lat)
    #print(df_eto_cfsv2)
    df_eto_concat= pd.concat([df_eto, df_eto_var_cfsv2],axis=0)
    #df_eto_his = check_data(start_date,end_date,df_eto)
    df_eto_his = check_data(start_date,end_date,df_eto_concat)
    #return df_eto_his
    return df_eto_his

def model_predict_precipitacion(roi):
    """This function is not being used (future dev)"""
    model_predict_precip = pickle.load(open('modelo_Precipitaciones.pkl','rb'))
    current_date = datetime.now().date() - timedelta(days=1)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=7)
    start_date= days_ago.strftime("%Y-%m-%d")
    precip_his = precipitation(start_date, end_date,roi)
    precip_his.dropna(inplace=True)
    precip_his.rename_axis("Date",inplace=True)
    precip_his=precip_his.asfreq('d')
    last_window = precip_his['Sensor_Values'].tail(3)
    #print(last_window)
    #print(type(last_window))
    steps=3
    prediction= model_predict_precip.predict(steps,last_window=last_window)
    return prediction

def load_model(filename):

    with pkg_resources.open_binary('eto_sm_module.models', filename) as f:
        return pickle.load(f)
    
def model_predict_sm(roi):
    """This function predicts sm"""
    #model_predict_sm = pickle.load(open('./models/modelo_humedad_suelo.pkl','rb'))
    model_predict_sm = load_model('modelo_humedad_suelo.pkl')
    current_date = datetime.now().date() - timedelta(days=5)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=18)
    start_date= days_ago.strftime("%Y-%m-%d")
    soilm_his = soil_moisturefun(start_date, end_date,roi)
    soilm_his.dropna(inplace=True)
    soilm_his.rename_axis("Date",inplace=True)
    soilm_his=soilm_his.asfreq('d')
    last_window = soilm_his['Sensor_Values'].tail(11)
    steps=8
    prediction= model_predict_sm.predict(steps,last_window=last_window)
    return prediction

def model_predict_eto(roi,lon,lat):
    """This function predicts eto"""
    #model_predict_eto = pickle.load(open('./models/modelo_ETo-2.pkl','rb'))
    model_predict_eto = load_model('modelo_ETo-2.pkl')
    current_date = datetime.now().date() - timedelta(days=4)
    end_date = current_date.strftime("%Y-%m-%d")
    days_ago = current_date - timedelta(days=31)
    start_date= days_ago.strftime("%Y-%m-%d")
    eto_his = eto_var(start_date, end_date,roi,lon,lat)
    eto_his.dropna(inplace=True)
    eto_his.rename_axis("Date",inplace=True)
    eto_his=eto_his.asfreq('d')
    last_window = eto_his['Sensor_Values'].tail(30)
    #last_window = last_window.fillna(last_window.mean())
    last_window = last_window.interpolate()
    steps=7
    prediction= model_predict_eto.predict(steps,last_window=last_window)
    return prediction

def create_app(account_gge:str,auth_file:str)-> FastAPI:
    """This function creates the apis"""

    app = FastAPI()
    @app.get("/")

    def index():
        return{"message" : "Completar la URL y campos"}

    @app.get("/satweather/")

    def get_sat_data(lat: float,lon: float):
        """This function creates api to get data"""
        service_account =account_gge
        credentials = ee.ServiceAccountCredentials(service_account, auth_file)
        ee.Initialize(credentials)

        # Region of interest (ROI)
        #roi = ee.Geometry.Point(-74.9840391, 5.1978685)  # Finca San Luis
        roi = ee.Geometry.Point(lon, lat)
        current_date = datetime.now().date() - timedelta(days=1)
        end_date = current_date.strftime("%Y-%m-%d")
        seven_days_ago = current_date - timedelta(days=10)
        start_date= seven_days_ago.strftime("%Y-%m-%d")
        precip_his = precipitation(start_date, end_date,roi)
        humedad_suelo_his = soil_moisturefun(start_date, end_date,roi)
        df_eto_his = df_eto_cfsv2(start_date, end_date,roi,lon,lat)
        eto_his = check_data(start_date,end_date,df_eto_his)
        #eto_his = eto_var(start_date, end_date,roi,lon,lat)
        #print(eto_his)

        dict_his={}
        for i, j in enumerate (precip_his.index):
            ts = str(j)
            precipitacion= precip_his['Sensor_Values'].loc[ts].round(3)
            humedadsuelo= humedad_suelo_his['Sensor_Values'].loc[ts].round(3)
            etovar = eto_his['Sensor_Values'].loc[ts].round(3)
            #print(etovar)
            if np.isnan(precipitacion):
                precipitacion = None
            if np.isnan(humedadsuelo):
                humedadsuelo = None
            if np.isnan(etovar):
                #print(etovar)
                etovar= None
            dict_his["Dato{}".format(i)]= {"ts":ts, "Precipitacion":precipitacion, "Humedad_Suelo":humedadsuelo, "ETo":etovar}

        return dict_his

    @app.get("/forecastWeather/")
    def modelos_predict(lat:float, lon:float):
        """This function creates api forecast"""

        service_account = account_gge
        credentials = ee.ServiceAccountCredentials(service_account, auth_file)
        ee.Initialize(credentials)
        roi = ee.Geometry.Point(lon, lat)
        #precipi_predict= model_predict_precipitacion(roi)
        soilm_predict=model_predict_sm(roi)
        eto_predict=model_predict_eto(roi,lon,lat)
        #print(soilm_predict)
        #print(eto_predict)
        dict_predict={}
        #current_day=datetime.now().date().strftime("%Y-%m-%d")
        i=0
        for j in (eto_predict.index):
            ts= str(j)
            #if ts >= current_day:
            eto_predict_value= eto_predict[ts]
            sm_predict_value= soilm_predict[ts]
            dict_predict["Dato{}".format(i)]={"ts":ts,"ETo_pred":eto_predict_value,"SM_pred":sm_predict_value}
            i+=1
        return dict_predict
    return app
