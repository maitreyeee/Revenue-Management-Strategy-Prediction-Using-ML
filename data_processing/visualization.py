import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import pandas as pd

def plot_forecast_errors(forecast_df, title):
    """
    Create a heatmap of forecast errors by booking class and timeframe.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data
        title (str): Plot title
    """
    # Calculate mean forecast errors
    fcst_error = forecast_df.groupby(['booking_class', 'timeframe'])[['Forecast Error']].mean().reset_index()
    
    # Create pivot table for heatmap
    error_pivot = fcst_error.pivot(index='booking_class', columns='timeframe', values='Forecast Error')
    
    # Create heatmap
    plt.figure(figsize=(8,5))
    sns.heatmap(error_pivot, cmap='PRGn', center=0, fmt='.1f')
    plt.title(title)
    plt.xlabel('Timeframe')
    plt.ylabel('Booking Class')
    plt.show()

def plot_feature_correlations(X_train, y_train):
    """
    Plot feature correlations with target variable.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    """
    # Calculate correlations
    correlations = pd.DataFrame()
    correlations['feature'] = X_train.columns
    correlations['correlation'] = [abs(X_train[col].corr(y_train)) for col in X_train.columns]
    correlations = correlations.sort_values('correlation', ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    plt.bar(correlations['feature'], correlations['correlation'])
    plt.xticks(rotation=90)
    plt.title('Feature Correlations with Target Variable')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(X_train):
    """
    Plot correlation matrix heatmap.
    
    Args:
        X_train (pd.DataFrame): Training features
    """
    corr_matrix = X_train.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_model_evaluation(model, X_test, y_test, feature_names=None):
    """
    Generate evaluation plots for a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
    """
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', cbar=False,
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
                annot_kws={"size": 18})
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend()
    plt.show()
    
    # SHAP Summary Plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.show()

def plot_win_loss_histogram(data, carrier, num_bins=10, strategy1='UEEE', strategy2='EEEE'):
    """
    Plot win/loss histogram for revenue differences.
    
    Args:
        data (np.array): Revenue difference data
        carrier (str): Carrier name
        num_bins (int): Number of histogram bins
        strategy1 (str): First strategy name
        strategy2 (str): Second strategy name
    """
    min_x = data.min()
    max_x = data.max()
    width = (max_x - min_x) / num_bins
    custom_bins = np.linspace(min_x, max_x, num_bins+1)
    
    pos_data, neg_data = data[data >= 0], data[data < 0]
    total_count = len(data)
    green_percent = 100 * len(pos_data) / total_count
    red_percent = 100 * len(neg_data) / total_count
    
    plt.hist(pos_data, bins=custom_bins, color='green', edgecolor='black', 
             label=f'Positive {strategy1}>{strategy2} - {green_percent:.2f}%')
    plt.hist(neg_data, bins=custom_bins, color='red', edgecolor='black', 
             label=f'Negative {strategy1}<{strategy2} - {red_percent:.2f}%')
    
    # Calculate and plot statistics
    mean = data.mean()
    std_dev = data.std()
    
    plt.axvline(x=mean, color='black', linestyle='--', label='Mean')
    plt.axvline(x=mean + std_dev, color='blue', linestyle='--', label='Mean ± 1 Std Dev')
    plt.axvline(x=mean - std_dev, color='blue', linestyle='--')
    plt.axvline(x=mean + 2 * std_dev, color='orange', linestyle='--', label='Mean ± 2 Std Dev')
    plt.axvline(x=mean - 2 * std_dev, color='orange', linestyle='--')
    
    plt.title(f'% Revenue Difference Histogram for {carrier}')
    plt.xlabel('Revenue % Difference')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_booking_window_errors(forecast_df, title):
    """
    Plot forecast errors by booking window.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data with booking window columns
        title (str): Plot title
    """
    # Prepare data for the plot
    window_data = {
        'Window 1': {
            'Positive': forecast_df['BkgWndw1_Pos'].mean(),
            'Negative': forecast_df['BkgWndw1_Neg'].mean()
        },
        'Window 2': {
            'Positive': forecast_df['BkgWndw2_Pos'].mean(),
            'Negative': forecast_df['BkgWndw2_Neg'].mean()
        },
        'Window 3': {
            'Positive': forecast_df['BkgWndw3_Pos'].mean(),
            'Negative': forecast_df['BkgWndw3_Neg'].mean()
        }
    }
    
    # Plot mean errors as grouped bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(window_data))
    width = 0.35
    
    pos_means = [data['Positive'] for data in window_data.values()]
    neg_means = [data['Negative'] for data in window_data.values()]
    
    plt.bar(x - width/2, pos_means, width, label='Positive Error', color='green', alpha=0.7)
    plt.bar(x + width/2, neg_means, width, label='Negative Error', color='red', alpha=0.7)
    
    # Add error bars showing standard deviation
    for i, window in enumerate(window_data.keys()):
        pos_std = forecast_df[f'BkgWndw{i+1}_Pos'].std()
        neg_std = forecast_df[f'BkgWndw{i+1}_Neg'].std()
        plt.errorbar(i - width/2, pos_means[i], yerr=pos_std, fmt='none', color='black', capsize=5)
        plt.errorbar(i + width/2, neg_means[i], yerr=neg_std, fmt='none', color='black', capsize=5)
    
    plt.xlabel('Booking Window')
    plt.ylabel('Mean Forecast Error')
    plt.title(title)
    plt.xticks(x, window_data.keys())
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_booking_class_errors(forecast_df, title):
    """
    Plot forecast errors by booking class.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data with booking class columns
        title (str): Plot title
    """
    # Prepare data for the plot
    class_data = {
        'Y0-Y3': {
            'Positive': forecast_df['BkgClsY0Y3_Pos'].mean(),
            'Negative': forecast_df['BkgClsY0Y3_Neg'].mean()
        },
        'Y4-Y6': {
            'Positive': forecast_df['BkgClsY4Y6_Pos'].mean(),
            'Negative': forecast_df['BkgClsY4Y6_Neg'].mean()
        },
        'Y7-Y9': {
            'Positive': forecast_df['BkgClsY7Y9_Pos'].mean(),
            'Negative': forecast_df['BkgClsY7Y9_Neg'].mean()
        }
    }
    
    # Plot mean errors as grouped bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_data))
    width = 0.35
    
    pos_means = [data['Positive'] for data in class_data.values()]
    neg_means = [data['Negative'] for data in class_data.values()]
    
    plt.bar(x - width/2, pos_means, width, label='Positive Error', color='green', alpha=0.7)
    plt.bar(x + width/2, neg_means, width, label='Negative Error', color='red', alpha=0.7)
    
    # Add error bars showing standard deviation
    for i, booking_class in enumerate(class_data.keys()):
        pos_std = forecast_df[f'BkgCls{booking_class.replace("-", "")}_Pos'].std()
        neg_std = forecast_df[f'BkgCls{booking_class.replace("-", "")}_Neg'].std()
        plt.errorbar(i - width/2, pos_means[i], yerr=pos_std, fmt='none', color='black', capsize=5)
        plt.errorbar(i + width/2, neg_means[i], yerr=neg_std, fmt='none', color='black', capsize=5)
    
    plt.xlabel('Booking Class')
    plt.ylabel('Mean Forecast Error')
    plt.title(title)
    plt.xticks(x, class_data.keys())
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pooled_std_dev_comparison(combined_data):
    """
    Plot comparison of pooled standard deviations between P and U strategies.
    
    Args:
        combined_data (pd.DataFrame): Combined dataset with pooled_fcst_std_dev_P and pooled_fcst_std_dev_U
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_data['pooled_fcst_std_dev_P'], combined_data['pooled_fcst_std_dev_U'], alpha=0.5)
    plt.plot([0, combined_data['pooled_fcst_std_dev_P'].max()], 
             [0, combined_data['pooled_fcst_std_dev_P'].max()], 
             'r--', label='Equal Line')
    plt.xlabel('Pooled Std Dev (P)')
    plt.ylabel('Pooled Std Dev (U)')
    plt.title('Comparison of Pooled Standard Deviations')
    plt.legend()
    plt.show() 