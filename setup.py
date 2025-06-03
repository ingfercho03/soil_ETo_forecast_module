import os
from setuptools import setup, find_packages

requirements_file = 'requirements.txt'

if os.path.isfile(requirements_file):
    with open(requirements_file, 'r') as f:
        required_dependencies = [line.strip() for line in f]
else:
    required_dependencies = []

setup(
    name="eto_sm_module",  
    version="0.1.0",  
    description="Module for Retrieving and Forecasting ETo and Soil Moisture Using Remote Sensing",  
    author="Diego Rodriguez",  
    author_email='diego.rodriguez-t@mail.escuelaing.edu.co',
    url='https://github.com/ingfercho03/soil_ETo_forecast_module.git',
    license="MIT",  
    packages=find_packages(),
    install_requires=required_dependencies,
    include_package_data=True,
    package_data={
        'eto_sm_module.models': ['model_sm.pkl','model_ETo-2.pkl'],
    },
)
