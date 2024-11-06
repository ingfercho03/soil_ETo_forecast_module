# Soil Moisture and Reference Evapotranspiration Forecasting Module

This repository contains a project focused on forecasting soil moisture and reference evapotranspiration (ETo) using satellite data. The project is structured into three main folders: datasets, notebooks, and thingsBoard. The forecasting module integrates these components to predict soil moisture and ETo, leveraging machine learning models and time series analysis. The final forecasting module is implemented in Python and can be executed via the main.py file.

Folder Structure

The repository is organized as follows:

1. datasets
This folder contains datasets related to soil moisture and reference evapotranspiration (ETo) based on satellite data. These datasets are used for training and validating machine learning models and performing time series analysis.

Contents:
Soil moisture data
Reference evapotranspiration (ETo) data

2. notebooks
This folder contains Jupyter notebooks that explore time series analysis techniques and machine learning models used for predicting soil moisture and ETo.

Contents:
Time series analysis: Notebooks that analyze historical data to identify patterns and trends in soil moisture and ETo.
Machine learning models: Notebooks that implement and evaluate different models to predict soil moisture and ETo based on historical data.

3. thingsBoard
This folder contains the configuration and scripts for integrating the forecasting module with the ThingsBoard platform. It includes dashboards, rule engines, and other components to visualize and manage the forecasting data.

Contents:
Dashboards: Visual interfaces for monitoring soil moisture and ETo predictions in real-time.
Rule Engine: Logic and configurations that allow for automatic actions based on the forecasted values.
Integration: Scripts and configuration files to integrate the forecasting module with the ThingsBoard platform.

4. main.py
The main.py file contains the Python code for the soil moisture and ETo forecasting module. This script integrates the data and the machine learning models to predict future values of soil moisture and ETo.

Dependencies
skforecast
ETo
ETo
ee
