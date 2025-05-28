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

├── LICENSE                          <- License file for the project.
├── requirements.txt                 <- Python dependencies required for the project.
├── setup.py                         <- Package setup file for installation.
├── README.md                        <- This comprehensive README file.
├── data/                            <- Stores datasets
├── docs/                            <- Sphinx documentation for the eto_sm_module.
├── models/                          <- Machine learning models.
├── notebooks/                       <- Jupyter notebooks for data exploration      time-series analysis and model training
└── eto_sm_module/                   <- Core project source code. This is the main Python package.
└── thingsboard/                     <- Configuration files for ThingsBoard integration.


## Getting Started

### Installation

To set up the project and install the necessary dependencies, please refer to the detailed installation instructions in our [Sphinx documentation](link-to-your-readthedocs-installation-section).

## License

This project is licensed under the [MIT License](LICENSE).

## References


* [ETo package: M. Kittridge, “Eto - a python package for calculating reference evapo-
transpiration,” 2018]
* [skforecast: https://skforecast.org/0.16.0/index.html]
* [Google Earth Engine Documentation: https://developers.google.com/earth-engine/guides]
* [ThingsBoard Documentation: https://thingsboard.io/docs/]

