import pandas as pd
import numpy as np

def load_carrier_history_data():
    """
    Load carrier history data for both PEEE and UEEE strategies.
    
    Returns:
        tuple: (PEEE_carrierhistory, UEEE_carrierhistory)
    """
    PEEE_carrierhistory = pd.read_csv('Data/PvsUdata/carrier_history_peee.csv')
    UEEE_carrierhistory = pd.read_csv('Data/PvsUdata/carrier_history_ueee.csv')
    
    # Filter for airline AL1
    PEEE_carrierhistory = PEEE_carrierhistory[PEEE_carrierhistory['carrier'] == 'AL1']
    UEEE_carrierhistory = UEEE_carrierhistory[UEEE_carrierhistory['carrier'] == 'AL1']
    
    # Fill NaN values
    PEEE_carrierhistory = PEEE_carrierhistory.fillna(0)
    UEEE_carrierhistory = UEEE_carrierhistory.fillna(0)
    
    return PEEE_carrierhistory, UEEE_carrierhistory

def load_forecast_data():
    """
    Load forecast data for both PEEE and UEEE strategies.
    
    Returns:
        tuple: (PEEE_forecast, UEEE_forecast)
    """
    PEEE_forecast = pd.read_csv('Data/PvsUdata/forecast_accuracy_peee.csv.gz', compression='gzip')
    UEEE_forecast = pd.read_csv('Data/PvsUdata/forecast_accuracy_ueee.csv.gz', compression='gzip')
    
    # Drop unnamed columns
    PEEE_forecast = PEEE_forecast.drop(columns=['Unnamed: 0'])
    UEEE_forecast = UEEE_forecast.drop(columns=['Unnamed: 0'])
    
    # Fill NaN values
    PEEE_forecast = PEEE_forecast.fillna(0)
    UEEE_forecast = UEEE_forecast.fillna(0)
    
    return PEEE_forecast, UEEE_forecast 