import pandas as pd
import numpy as np
from data_processing.data_loader import load_carrier_history_data, load_forecast_data
from data_processing.data_preprocessor import (
    calculate_forecast_metrics,
    group_forecast_data,
    prepare_features_for_ml,
    select_features,
    carrier_history_metrics,
    merge_carrier_history_and_forecast,
    combine_p_and_u_datasets
)
from data_processing.visualization import (
    plot_forecast_errors,
    plot_feature_correlations,
    plot_correlation_matrix,
    plot_win_loss_histogram,
    plot_booking_window_errors,
    plot_booking_class_errors,
    plot_pooled_std_dev_comparison
)
from data_processing.ml_analysis import (
    train_random_forest,
    train_logistic_regression,
    train_xgboost,
    evaluate_model,
    plot_model_results,
    save_model_results,
    compare_models,
    plot_model_comparison
)

def main():
    # Load data
    print("Loading data...")
    PEEE_carrierhistory, UEEE_carrierhistory = load_carrier_history_data()
    PEEE_forecast, UEEE_forecast = load_forecast_data()
    
    # Calculate carrier history metrics
    print("Calculating carrier history metrics...")
    PEEE_carrierhistory = carrier_history_metrics(PEEE_carrierhistory)
    UEEE_carrierhistory = carrier_history_metrics(UEEE_carrierhistory)

    # Calculate forecast metrics
    print("Calculating forecast metrics...")
    PEEE_forecast = calculate_forecast_metrics(PEEE_forecast)
    UEEE_forecast = calculate_forecast_metrics(UEEE_forecast)
    
    # Group forecast data
    print("Grouping forecast data...")
    grouped_PEEE = group_forecast_data(PEEE_forecast)
    grouped_UEEE = group_forecast_data(UEEE_forecast)
    
    # Merge carrier history and forecast data
    print("Merging carrier history and forecast data...")
    PEEE_merged = merge_carrier_history_and_forecast(PEEE_carrierhistory, grouped_PEEE)
    UEEE_merged = merge_carrier_history_and_forecast(UEEE_carrierhistory, grouped_UEEE)
    
    # Combine P and U datasets
    print("Combining P and U datasets...")
    combined_data = combine_p_and_u_datasets(PEEE_merged, UEEE_merged)

    print(combined_data.head())

    # Prepare features for machine learning
    print("Preparing features for machine learning...")
    X_train, X_test, y_train, y_test = prepare_features_for_ml(combined_data)

    # Select features
    print("Selecting features...")
    selected_features = select_features(X_train, y_train)

    print(selected_features)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Save processed data
    print("Saving processed data...")
    X_train.to_csv('Data/PvsUdata/X_train_resampled.csv', index=False)
    X_test.to_csv('Data/PvsUdata/X_test.csv', index=False)
    y_train.to_csv('Data/PvsUdata/Y_train_resampled.csv', index=False)
    y_test.to_csv('Data/PvsUdata/Y_test.csv', index=False)
    
    # Generate data visualizations
    # print("Generating data visualizations...")
    
    # # Forecast error visualizations
    # plot_forecast_errors(PEEE_forecast, 'Forecast Errors AL1 (PEEE)')
    # plot_forecast_errors(UEEE_forecast, 'Forecast Errors AL1 (UEEE)')
    
    # # Booking window and class error visualizations
    # plot_booking_window_errors(PEEE_forecast, 'Booking Window Errors AL1 (PEEE)')
    # plot_booking_window_errors(UEEE_forecast, 'Booking Window Errors AL1 (UEEE)')
    # plot_booking_class_errors(PEEE_forecast, 'Booking Class Errors AL1 (PEEE)')
    # plot_booking_class_errors(UEEE_forecast, 'Booking Class Errors AL1 (UEEE)')
    
    # # Feature analysis visualizations
    # plot_feature_correlations(X_train, y_train)
    # plot_correlation_matrix(X_train)
    
    # # Revenue and standard deviation comparisons
    # plot_win_loss_histogram(
    #     (combined_data['revenue_diff_U-P']*100/combined_data['revenue_P']),
    #     carrier='AL1',
    #     num_bins=15,
    #     strategy1='U',
    #     strategy2='P'
    # )
    # plot_pooled_std_dev_comparison(combined_data)
    
    # Machine Learning Analysis
    print("\nStarting Machine Learning Analysis...")
    
    # Train models
    print("Training models...")
    models = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf_model, rf_params = train_random_forest(X_train, y_train)
    models['Random Forest'] = (rf_model, rf_params)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = (lr_model, lr_params)
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    models['XGBoost'] = (xgb_model, xgb_params)
    
    # Compare models
    print("Comparing models...")
    results_df = compare_models(models, X_test, y_test)
    plot_model_comparison(results_df)
    
    # Save results for best model
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model, best_params = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best parameters: {best_params}")
    
    # Evaluate best model
    print("Evaluating best model...")
    y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test)
    
    # Generate model visualizations
    print("Generating model visualizations...")
    feature_importance = plot_model_results(y_test, y_pred, y_pred_proba, best_model, X_train)
    
    # Save model results
    print("Saving model results...")
    save_model_results(best_model, y_test, y_pred, feature_importance, best_params)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 