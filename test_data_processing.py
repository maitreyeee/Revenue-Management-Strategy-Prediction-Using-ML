import pandas as pd
import numpy as np
from data_processing.data_loader import load_carrier_history_data, load_forecast_data
from data_processing.data_preprocessor import (
    calculate_forecast_metrics,
    group_forecast_data,
    prepare_features_for_ml,
    calculate_pooled_std_dev,
    combine_p_and_u_datasets,
    carrier_history_metrics,
    merge_carrier_history_and_forecast
)

def print_data_info(df, name):
    """Helper function to print data information"""
    print(f"\n{name} Data Structure:")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    print("\nFirst few rows:")
    print(df.head())
    print("=" * 50)

def test_full_pipeline():
    """Test the entire data preprocessing pipeline using actual data"""
    print("\nTesting full data preprocessing pipeline...")
    try:
        # Load actual data
        print("\nLoading data...")
        PEEE_forecast, UEEE_forecast = load_forecast_data()
        PEEE_carrierhistory, UEEE_carrierhistory = load_carrier_history_data()
        
        # Take a small subset of the data (first 100 rows)
        PEEE_subset = PEEE_forecast.head(1000).copy()
        UEEE_subset = UEEE_forecast.head(1000).copy()
        PEEE_carrier_subset = PEEE_carrierhistory.head(1000).copy()
        UEEE_carrier_subset = UEEE_carrierhistory.head(1000).copy()
        
        print_data_info(PEEE_subset, "PEEE Forecast Subset")
        print_data_info(UEEE_subset, "UEEE Forecast Subset")
        print_data_info(PEEE_carrier_subset, "PEEE Carrier History Subset")
        print_data_info(UEEE_carrier_subset, "UEEE Carrier History Subset")
        
        # Step 1: Calculate carrier history metrics
        print("\nCalculating carrier history metrics...")
        PEEE_carrier_metrics = carrier_history_metrics(PEEE_carrier_subset)
        UEEE_carrier_metrics = carrier_history_metrics(UEEE_carrier_subset)
        
        print_data_info(PEEE_carrier_metrics, "PEEE Carrier Metrics")
        print_data_info(UEEE_carrier_metrics, "UEEE Carrier Metrics")
        
        # Verify carrier history metrics
        assert 'Price/Yield' in PEEE_carrier_metrics.columns, "Missing Price/Yield column in carrier history"
        
        # Step 2: Calculate forecast metrics
        print("\nCalculating forecast metrics...")
        PEEE_metrics = calculate_forecast_metrics(PEEE_subset)
        UEEE_metrics = calculate_forecast_metrics(UEEE_subset)
        
        print_data_info(PEEE_metrics, "PEEE Metrics")
        print_data_info(UEEE_metrics, "UEEE Metrics")
        
        # Verify forecast metrics calculations
        assert 'Forecast Error' in PEEE_metrics.columns, "Missing Forecast Error column"
        assert 'Forecast Error Positive' in PEEE_metrics.columns, "Missing Forecast Error Positive column"
        assert 'Forecast Error Negative' in PEEE_metrics.columns, "Missing Forecast Error Negative column"
        assert 'booking_window' in PEEE_metrics.columns, "Missing booking_window column"
        assert 'Booking Class' in PEEE_metrics.columns, "Missing Booking Class column"
        
        # Step 3: Group forecast data
        print("\nGrouping forecast data...")
        PEEE_grouped = group_forecast_data(PEEE_metrics)
        UEEE_grouped = group_forecast_data(UEEE_metrics)
        
        print_data_info(PEEE_grouped, "PEEE Grouped")
        print_data_info(UEEE_grouped, "UEEE Grouped")
        
        # Verify grouping and pooled standard deviation calculation
        assert 'pooled_fcst_std_dev' in PEEE_grouped.columns, "Missing pooled_fcst_std_dev column"
        
        # Calculate pooled standard deviation manually to verify
        def verify_pooled_std(group, metrics_df):
            group_data = metrics_df[(metrics_df['trial'] == group['trial']) & 
                                  (metrics_df['sample'] == group['sample'])]
            expected_std = calculate_pooled_std_dev(group_data['fcst_std_dev'], group_data['num_obs'])
            actual_std = group['pooled_fcst_std_dev']
            assert abs(expected_std - actual_std) < 1e-10, f"Pooled standard deviation calculation incorrect for trial {group['trial']}, sample {group['sample']}"
        
        # Verify pooled standard deviation for each group
        PEEE_grouped.apply(lambda x: verify_pooled_std(x, PEEE_metrics), axis=1)
        UEEE_grouped.apply(lambda x: verify_pooled_std(x, UEEE_metrics), axis=1)
        
        # Step 4: Merge carrier history and forecast data
        print("\nMerging carrier history and forecast data...")
        PEEE_merged = merge_carrier_history_and_forecast(PEEE_carrier_metrics, PEEE_grouped)
        UEEE_merged = merge_carrier_history_and_forecast(UEEE_carrier_metrics, UEEE_grouped)
        
        print_data_info(PEEE_merged, "PEEE Merged")
        print_data_info(UEEE_merged, "UEEE Merged")
        
        # Step 5: Combine P and U datasets
        print("\nCombining P and U datasets...")
        combined_data = combine_p_and_u_datasets(PEEE_merged, UEEE_merged)
        
        print_data_info(combined_data, "Combined Dataset")
        
        # Verify Price/Yield ratio calculation
        assert 'Price/Yield_P' in combined_data.columns, "Missing Price/Yield_P column"
        assert 'Price/Yield_U' in combined_data.columns, "Missing Price/Yield_U column"
        
        # Verify combined dataset
        required_columns = [
            'trial', 'sample', 'revenue_P', 'revenue_U', 'revenue_diff_U-P',
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
        
        assert all(col in combined_data.columns for col in required_columns), "Missing required columns in combined dataset"
        
        # Step 6: Prepare features for ML
        print("\nPreparing features for ML...")
        X_train, X_test, y_train, y_test = prepare_features_for_ml(combined_data)
        
        print_data_info(X_train, "X_train")
        print_data_info(X_test, "X_test")
        print("\ny_train values:", y_train.value_counts())
        print("y_test values:", y_test.value_counts())
        
        # Verify ML preparation
        assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
        assert isinstance(y_train, pd.Series), "y_train should be a Series"
        assert isinstance(y_test, pd.Series), "y_test should be a Series"
        assert len(X_train) == len(y_train), "X_train and y_train should have same length"
        assert len(X_test) == len(y_test), "X_test and y_test should have same length"
        
        print("\nPipeline Summary:")
        print(f"Original PEEE data shape: {PEEE_subset.shape}")
        print(f"Original UEEE data shape: {UEEE_subset.shape}")
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        print("\n✓ Full pipeline test passed")
        return True
    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Starting data processing function tests...")
    
    # Run the full pipeline test
    test_full_pipeline()
    
    print("\nTest Summary:")
    print("Full pipeline test completed. Check the output above for details.")

if __name__ == "__main__":
    main()