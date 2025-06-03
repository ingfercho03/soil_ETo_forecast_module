# ETo and Soil Moisture Retrieval and Forecasting Module

This module provides functionalities for retrieving reference evapotranspiration (ETo) and soil moisture (SM) data from satellite products, and for forecasting these parameters using machine learning models. The primary goal is to support accurate crop irrigation needs assessment.

## Features

* **Satellite Data Retrieval:** Access historical ETo and SM data as time series from satellite products.

* **Machine Learning Forecasting:** Predict future ETo and SM values using integrated machine learning models. The repository includes two pre-trained machine learning models used for forecasting key agroclimatic variables:

| **File Name**         | **Description**                                                                 |
|-----------------------|---------------------------------------------------------------------------------|
| `model_ETo-2.pkl`     | Forecasts **reference evapotranspiration (ET₀)** based on weather and climate inputs. |
| `model_sm.pkl`        | Predicts **soil moisture levels** using historical and forecasted weather data.       |

* **API Interface:** Exposes data retrieval and forecasting functionalities via a RESTful API for easy integration.

* **ThingsBoard Integration:** Designed to facilitate data visualization and irrigation water requirement calculations within the ThingsBoard IoT platform.

## Installation

> **Note:** This module was tested on **Ubuntu Linux 64-bit**. Compatibility with other operating systems has not been extensively verified.


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

This module can be integrated with the ThingsBoard IoT platform to visualize real-time and forecasted data on customizable dashboards, as well as to generate irrigation water requirement tables.

To streamline integration, we provide two Rule Chain JSON files and two Dashboard JSON files. These are located in the Thingsboard/ folder of the repository:

### 📁 Provided Configuration Files


| **Type**     | **File Name**                            | **Description**                                           |
|--------------|-------------------------------------------|-----------------------------------------------------------|
| Rule Chain   | `rule_chain_get_satellite_data.json`      | Fetches satellite-based agroclimatic data                |
| Rule Chain   | `rule_chain_forecast_eto_sm_data.json`    | Retrieves forecasted evapotranspiration (ET₀) and soil moisture data |
| Dashboard    | `dashboard_climate_data.json`             | Displays climate and satellite-based data                |
| Dashboard    | `dashboard_irrigation_forecast.json`      | Shows forecasted irrigation needs and charts             |


### Step-by-Step Integration with ThingsBoard

1. Import the Rule Chains

    Log in to your ThingsBoard instance as a Tenant Administrator.

    Navigate to Rule Chains from the left-hand menu.

    Click the "Import Rule Chain" button (folder icon at the top).

    Upload the following files from the Thingsboard/ directory:

        rule_chain_get_satellite_data.json

        rule_chain_forecast_eto_sm_data.json

    Activate the rule chains as needed.

2. Import the Dashboards

    Go to the Dashboards section in ThingsBoard.

    Click the "+" icon and select "Import Dashboard".

    Upload:

        dashboard_climate_data.json

        dashboard_irrigation_forecast.json

    Assign the dashboards to relevant devices or assets if required.

3. REST API Integration via Rule Engine

Within the imported rule chains, "REST API Call Nodes" are used to pull data from your module:

    Satellite Weather Endpoint: /satweather

    Forecast Weather & Irrigation Data: /forecastWeather

These API nodes are pre-configured but should be reviewed to:

    Set the correct base URL for your hosted API server

    Include any necessary authentication headers or tokens






Note: This documentation assumes basic familiarity with Python, virtual environments, and REST APIs. For detailed information on Google Earth Engine authentication, please refer to the official GEE documentation.
