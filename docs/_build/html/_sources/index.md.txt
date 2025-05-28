# ETo and Soil Moisture Retrieval and Forecasting Module

This module provides functionalities for retrieving reference evapotranspiration (ETo) and soil moisture (SM) data from satellite products, and for forecasting these parameters using machine learning models. The primary goal is to support accurate crop irrigation needs assessment.

## Features

* **Satellite Data Retrieval:** Access historical ETo and SM data as time series from satellite products.
* **Machine Learning Forecasting:** Predict future ETo and SM values using integrated machine learning models.
* **API Interface:** Exposes data retrieval and forecasting functionalities via a RESTful API for easy integration.
* **ThingsBoard Integration:** Designed to facilitate data visualization and irrigation water requirement calculations within the ThingsBoard IoT platform.

## Installation

To get started with the ETo and Soil Moisture module, follow these steps:

### 1. Create a Python Virtual Environment

It is highly recommended to install the module within a dedicated Python virtual environment to avoid conflicts with other Python projects.

```{code-block}
---
emphasize-lines: 1
---
python -m venv eto_sm_module_test
source eto_sm_module_test/bin/activate
```

### 2. Install the Module

Install the module directly from its GitHub repository:
```{code-block}
---
emphasize-lines: 1
---
pip install git+https://github.com/ingfercho03/soil_ETo_forecast_module.git
```
```{warning}
If you encounter an error related to `wheel` during installation, please install the `wheel` package within your virtual environment first
```
## Usage

### 1. Execute the Module and Start the API Server

To run the module and start the API server, you need to provide your Google Earth Engine (GEE) service account credentials.

```{code-block}
---
emphasize-lines: 1
---
python -m eto_sm_module --account_gge="your-service-account@ee-xxx.iam.gserviceaccount.com" --auth_file="your-auth.json"
```

    Replace "your-service-account@ee-xxx.iam.gserviceaccount.com" with your actual Google Earth Engine service account email.
    Replace "your-auth.json" with the path to your Google Earth Engine private key JSON file.

Once executed, the API server will typically run on http://127.0.0.1:8000.

### 2. Accessing the API Documentation

After the server is running, you can access the interactive API documentation (Swagger UI) at:

```{code-block}
---
emphasize-lines: 1
---
http://127.0.0.1:8000/docs
```
### 3. API Endpoints

The module exposes the following key API endpoints:

a. /satweather - Retrieve Historical Satellite Data

This API allows you to retrieve historical time-series data for reference evapotranspiration (ETo) and soil moisture (SM) based on satellite products for a given geographical location.

```{code-block}
---
emphasize-lines: 1
---
http://127.0.0.1:8000/satweather/?lat={your latitude}&lon={your longitude}

```

    Replace {your latitude} and {your longitude} with the desired geographical coordinates.

b. /forecastWeather - Retrieve Forecasted Data

This API provides predictions of reference evapotranspiration (ETo) and soil moisture (SM) using machine learning models for a specified location.

```{code-block}
---
emphasize-lines: 1
---
http://127.0.0.1:8000/forecastWeather/?lat={your latitude}&lon={your longitude}
```

    Replace {your latitude} and {your longitude} with the desired geographical coordinates.

## ThingsBoard Usage Scenario

This module can be integrated with the ThingsBoard IoT platform (https://thingsboard.io/) to visualize the retrieved and forecasted data on dashboards and to generate tables of irrigation water requirements.

To connect ThingsBoard with this module, follow these steps:
### 1. Configure ThingsBoard Rule Engine

Create a Rule Chain in ThingsBoard as illustrated in the figure below. This typically involves setting up nodes to interact with external REST APIs.

![Rule engine and nodes ](thingsboard_rule.png)

### 2. Invoke APIs from "REST API Call Node"

Within the "REST API Call Node" in your ThingsBoard Rule Chain, configure it to invoke the appropriate API endpoints provided by this module (e.g., /satweather or /forecastWeather) to fetch the required data. Refer to the ThingsBoard documentation for detailed instructions on configuring REST API calls.







Note: This documentation assumes basic familiarity with Python, virtual environments, and REST APIs. For detailed information on Google Earth Engine authentication, please refer to the official GEE documentation.
