# Satellite-Based Retrieval and Machine Learning Forecasting of Reference Evapotranspiration and Soil Moisture for Crop Irrigation Needs

## Project Overview

This repository hosts the code and resources for a comprehensive module designed to retrieve historical reference evapotranspiration (ETo) and soil moisture (SM) data from satellite products, and to forecast these critical agricultural parameters using advanced machine learning models. The primary objective is to provide timely and accurate insights to optimize crop irrigation strategies, thereby promoting efficient water use and sustainable agriculture.

## Features

* **Satellite Data Integration:** Seamlessly retrieves ETo and SM data base on satellite products.
* **Machine Learning-Powered Forecasting:** Leverages machine learning models to predict future ETo and SM values.
* **Time-Series Analysis:** Includes notebooks for analyzing historical ETo and SM time series.
* **RESTful API:** Exposes core functionalities via a user-friendly API, enabling easy integration with other systems.
* **ThingsBoard Integration:** Facilitates visualization and management of ETo, SM, and calculated irrigation requirements within the ThingsBoard IoT platform.

## Repository Structure

    ├── LICENSE            <- License file for the project.
    ├── README.md          <- This comprehensive README file.
    ├── data               <- Datasets.
    │
    ├── docs               <- Sphinx documentation for the eto_sm_module.
    │
    ├── notebooks          <- Jupyter notebooks for data exploration time-series analysis and model training
    │
    ├── requirements.txt   <- Python dependencies required for the project.
    │
    ├── setup.py           <- Package setup file for installation.
    │
    └── eto_sm_module      <- Core project source code. This is the main Python package.
        ├── models                <- Machine learning models.

            ├── model_Eto-2.pkl   <- Forecasts reference evapotranspiration (ET₀)   based on weather and climate inputs.

            ├── model_sm.pkl      <- Predicts soil moisture levels using historical and forecasted weather data.
    │
    └── thingsboard        <- Configuration files for ThingsBoard integration.  


## Getting Started

### Installation

To set up the project and install the necessary dependencies, please refer to the detailed installation instructions in our [Sphinx documentation](https://soil-eto-forecast-module.readthedocs.io/en/latest/).

## License

This project is licensed under the [MIT License](LICENSE).

## References


* [ETo package: M. Kittridge, “Eto - a python package for calculating reference evapo-
transpiration,” 2018]
* [skforecast: https://skforecast.org/0.16.0/index.html]
* [Google Earth Engine Documentation: https://developers.google.com/earth-engine/guides]
* [ThingsBoard Documentation: https://thingsboard.io/docs/]

