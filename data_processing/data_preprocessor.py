import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def calculate_pooled_std_dev(std_devs, counts):
    """
    Calculate pooled standard deviation for forecast data.
    
    Args:
        std_devs (pd.Series): Series of standard deviations
        counts (pd.Series): Series of corresponding counts (number of observations)
        
    Returns:
        float: Pooled standard deviation
    """
    dof = counts - 1  # degrees of freedom
    weighted_variance = np.sum(dof * (std_devs ** 2))
    total_dof = np.sum(dof)
    return np.sqrt(weighted_variance / total_dof) if total_dof > 0 else 0

def carrier_history_metrics(carrier_history_df):
    """
    Calculate carrier history metrics for a given carrier history dataframe.
    """
    # Calculate carrier history metrics
    carrier_history_df['Price/Yield'] = carrier_history_df['sold_priceable'] / carrier_history_df['sold']

    return carrier_history_df

def calculate_forecast_metrics(forecast_df):
    """
    Calculate forecast error metrics for a given forecast dataframe.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data
        
    Returns:
        pd.DataFrame: DataFrame with calculated metrics
    """
    # Calculate forecast error
    forecast_df['Forecast Error'] = forecast_df['fcst_mean'] - forecast_df['sold']
    
    # Calculate positive and negative forecast errors
    forecast_df['Forecast Error Positive'] = np.abs(np.where(forecast_df['Forecast Error'] > 0, 
                                                           forecast_df['Forecast Error'], 0))
    forecast_df['Forecast Error Negative'] = np.abs(np.where(forecast_df['Forecast Error'] < 0, 
                                                           forecast_df['Forecast Error'], 0))
    
    # Calculate booking window metrics
    forecast_df['booking_window'] = np.where(forecast_df['timeframe'] <= 5, 1, 
                                           np.where(forecast_df['timeframe'] <= 11, 2, 3))
    
    # Create booking window specific error columns
    forecast_df['BkgWndw1_Pos'] = np.where((forecast_df['booking_window'] == 1), 
                                          forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgWndw1_Neg'] = np.where((forecast_df['booking_window'] == 1), 
                                          forecast_df['Forecast Error Negative'], 0)
    forecast_df['BkgWndw2_Pos'] = np.where((forecast_df['booking_window'] == 2), 
                                          forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgWndw2_Neg'] = np.where((forecast_df['booking_window'] == 2), 
                                          forecast_df['Forecast Error Negative'], 0)
    forecast_df['BkgWndw3_Pos'] = np.where((forecast_df['booking_window'] == 3), 
                                          forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgWndw3_Neg'] = np.where((forecast_df['booking_window'] == 3), 
                                          forecast_df['Forecast Error Negative'], 0)
    
    # Calculate booking class metrics
    forecast_df['Booking Class'] = np.where(
        forecast_df['booking_class'].isin(['Y0', 'Y1', 'Y2', 'Y3']), 'Y0-Y3',
        np.where(forecast_df['booking_class'].isin(['Y4', 'Y5', 'Y6']), 'Y4-Y6',
                np.where(forecast_df['booking_class'].isin(['Y7', 'Y8', 'Y9']), 'Y7-Y9', 'OTHER')))
    
    # Create booking class specific error columns
    forecast_df['BkgClsY0Y3_Pos'] = np.where((forecast_df['Booking Class'] == 'Y0-Y3'), 
                                            forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgClsY0Y3_Neg'] = np.where((forecast_df['Booking Class'] == 'Y0-Y3'), 
                                            forecast_df['Forecast Error Negative'], 0)
    forecast_df['BkgClsY4Y6_Pos'] = np.where((forecast_df['Booking Class'] == 'Y4-Y6'), 
                                            forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgClsY4Y6_Neg'] = np.where((forecast_df['Booking Class'] == 'Y4-Y6'), 
                                            forecast_df['Forecast Error Negative'], 0)
    forecast_df['BkgClsY7Y9_Pos'] = np.where((forecast_df['Booking Class'] == 'Y7-Y9'), 
                                            forecast_df['Forecast Error Positive'], 0)
    forecast_df['BkgClsY7Y9_Neg'] = np.where((forecast_df['Booking Class'] == 'Y7-Y9'), 
                                            forecast_df['Forecast Error Negative'], 0)
    
    # Calculate pooled standard deviation if std_dev and num_obs columns exist
    if 'std_dev' in forecast_df.columns and 'num_obs' in forecast_df.columns:
        forecast_df['Pooled Std Dev'] = calculate_pooled_std_dev(
            forecast_df['std_dev'], 
            forecast_df['num_obs']
        )
    
    return forecast_df

def group_forecast_data(forecast_df):
    """
    Group forecast data by trial and sample.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data with calculated metrics
        
    Returns:
        pd.DataFrame: Grouped forecast data
    """
    # First calculate pooled standard deviation for each group
    grouped_std_dev = forecast_df.groupby(['trial', 'sample']).apply(
        lambda x: calculate_pooled_std_dev(x['fcst_std_dev'], x['num_obs'])
    ).reset_index(name='pooled_fcst_std_dev')
    
    # Then perform the main aggregation
    grouped = forecast_df.groupby(['trial', 'sample']).agg({
        'fcst_mean': 'sum',
        'Forecast Error': 'mean',
        'Forecast Error Positive': 'mean',
        'Forecast Error Negative': 'mean',
        'BkgWndw1_Pos': 'mean',
        'BkgWndw1_Neg': 'mean',
        'BkgWndw2_Pos': 'mean',
        'BkgWndw2_Neg': 'mean',
        'BkgWndw3_Pos': 'mean',
        'BkgWndw3_Neg': 'mean',
        'BkgClsY0Y3_Pos': 'mean',
        'BkgClsY0Y3_Neg': 'mean',
        'BkgClsY4Y6_Pos': 'mean',
        'BkgClsY4Y6_Neg': 'mean',
        'BkgClsY7Y9_Pos': 'mean',
        'BkgClsY7Y9_Neg': 'mean'
    }).reset_index()
    
    # Merge the pooled standard deviation with the main aggregation
    grouped = pd.merge(grouped, grouped_std_dev, on=['trial', 'sample'])
    
    return grouped

def merge_carrier_history_and_forecast(carrier_history_df, forecast_df):
    """
    Merge carrier history and forecast dataframes.
    """
    # Merge carrier history and forecast dataframes
    merged_df = pd.merge(carrier_history_df, forecast_df, on=['trial', 'sample'])
    return merged_df

def combine_p_and_u_datasets(p_df, u_df):
    """
    Combine P and U datasets with proper column naming.
    
    Args:
        p_df (pd.DataFrame): P dataset
        u_df (pd.DataFrame): U dataset
        
    Returns:
        pd.DataFrame: Combined dataset with proper column naming
    """
    # Ensure both dataframes have the same columns
    common_columns = ['trial', 'sample', 'sold', 'revenue', 'fcst_mean', 'Forecast Error', 
                     'Forecast Error Positive', 'Forecast Error Negative',
                     'BkgWndw1_Pos', 'BkgWndw1_Neg', 'BkgWndw2_Pos', 'BkgWndw2_Neg',
                     'BkgWndw3_Pos', 'BkgWndw3_Neg', 'BkgClsY0Y3_Pos', 'BkgClsY0Y3_Neg',
                     'BkgClsY4Y6_Pos', 'BkgClsY4Y6_Neg', 'BkgClsY7Y9_Pos', 'BkgClsY7Y9_Neg',
                     'Price/Yield', 'pooled_fcst_std_dev']
    
    # Create copies with only common columns
    p_df = p_df[common_columns].copy()
    u_df = u_df[common_columns].copy()
    
    # Rename columns with suffixes in one operation
    p_columns = {col: f'{col}_P' for col in p_df.columns if col not in ['trial', 'sample']}
    u_columns = {col: f'{col}_U' for col in u_df.columns if col not in ['trial', 'sample']}
    
    p_df = p_df.rename(columns=p_columns)
    u_df = u_df.rename(columns=u_columns)
    
    # Merge dataframes
    combined_df = pd.merge(p_df, u_df, on=['trial', 'sample'])
    
    # Calculate revenue difference in one operation
    combined_df['revenue_diff_U-P'] = combined_df['revenue_U'] - combined_df['revenue_P']
    
    return combined_df

def prepare_features_for_ml(data):
    """
    Prepare features for machine learning.
    
    Args:
        data (pd.DataFrame): Combined dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Define required columns
    required_columns = [
        'Price/Yield_P', 'Price/Yield_U', 'fcst_mean_P', 'fcst_mean_U',
        'Forecast Error_P', 'Forecast Error_U', 'Forecast Error Positive_P',
        'Forecast Error Positive_U', 'Forecast Error Negative_P',
        'Forecast Error Negative_U', 'pooled_fcst_std_dev_P',
        'pooled_fcst_std_dev_U', 'BkgWndw1_Pos_P', 'BkgWndw1_Pos_U',
        'BkgWndw1_Neg_P', 'BkgWndw1_Neg_U', 'BkgWndw2_Pos_P', 'BkgWndw2_Pos_U',
        'BkgWndw2_Neg_P', 'BkgWndw2_Neg_U', 'BkgWndw3_Pos_P', 'BkgWndw3_Pos_U',
        'BkgWndw3_Neg_P', 'BkgWndw3_Neg_U', 'BkgClsY0Y3_Pos_P',
        'BkgClsY0Y3_Pos_U', 'BkgClsY0Y3_Neg_P', 'BkgClsY0Y3_Neg_U',
        'BkgClsY4Y6_Pos_P', 'BkgClsY4Y6_Pos_U', 'BkgClsY4Y6_Neg_P',
        'BkgClsY4Y6_Neg_U', 'BkgClsY7Y9_Pos_P', 'BkgClsY7Y9_Pos_U',
        'BkgClsY7Y9_Neg_P', 'BkgClsY7Y9_Neg_U'
    ]
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create features and target
    X = data[required_columns].copy()
    y = (data['revenue_U'] > data['revenue_P']).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Transform numerical features
    pt = PowerTransformer()
    X_train = pd.DataFrame(pt.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(pt.transform(X_test), columns=X_test.columns)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def select_features(X_train, y_train, correlation_threshold=0.95):
    """
    Select features based on correlation analysis.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        correlation_threshold (float): Threshold for feature correlation
        
    Returns:
        list: Selected feature names
    """
    # Get correlation matrix between features
    corr_matrix = X_train.corr().abs()
    correlations = pd.DataFrame()
    correlations['feature'] = X_train.columns
    correlations['correlation'] = [abs(X_train[col].corr(y_train)) for col in X_train.columns]
    correlations = correlations.sort_values('correlation', ascending=False)

    # Find highly correlated feature pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    features_to_drop = []
    
    for col in upper.columns:
        highly_corr = upper[col][upper[col] > correlation_threshold].index.tolist()
        if highly_corr:
            for feat in highly_corr:
                corr_col = correlations[correlations['feature'] == col]['correlation'].iloc[0]
                corr_feat = correlations[correlations['feature'] == feat]['correlation'].iloc[0]
                if corr_col < corr_feat:
                    features_to_drop.append(col)
                else:
                    features_to_drop.append(feat)
    
    # Remove duplicate features to drop
    features_to_drop = list(set(features_to_drop))
    
    # Get final feature list
    selected_features = [col for col in X_train.columns if col not in features_to_drop]
    
    return selected_features 