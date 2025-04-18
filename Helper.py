# Description: This file contains the functions to run the simulation and save the output in the desired format.
# import
import os
import pandas as pd
import numpy as np
import passengersim as pax
from passengersim.utils.codeview import show_file
import time
from math import ceil, floor, log10
import matplotlib.pyplot as plt
import shap


# function to run the simulation on the YAML files
def run_simulation(cfg):

    # sim = pax.Simulation(cfg)
    sim = pax.MultiSimulation(cfg)
    
    summary = sim.run()

    formatted_summary = simOut(summary)
    # summary.cnx.close()
    del summary.cnx

    return formatted_summary

# function to take in the parameters and create a cfg file

def params_cfg(params, cfg):
    """
    Update the cfg file based on the provided parameters.

    Args:
    - params (dict): A dictionary containing the parameter values.
    - cfg (object): The configuration object to be updated.

    Returns:
    - cfg (object): The updated configuration object.
    """

    cfg.choice_models['business'].emult = params.get('choice_models__business__emult', cfg.choice_models['business'].emult)
    cfg.choice_models['leisure'].emult = params.get('choice_models__leisure__emult', cfg.choice_models['leisure'].emult)
    cfg.choice_models['business'].r1 = params.get('choice_models__business__r1', cfg.choice_models['business'].r1)
    cfg.choice_models['leisure'].r1 = params.get('choice_models__leisure__r1', cfg.choice_models['leisure'].r1)
    cfg.choice_models['business'].r2 = params.get('choice_models__business__r2', cfg.choice_models['business'].r2)
    cfg.choice_models['leisure'].r2 = params.get('choice_models__leisure__r2', cfg.choice_models['leisure'].r2)
    cfg.choice_models['business'].r3 = params.get('choice_models__business__r3', cfg.choice_models['business'].r3)
    cfg.choice_models['leisure'].r3 = params.get('choice_models__leisure__r3', cfg.choice_models['leisure'].r3)
    
    return cfg


# %%
def flatten_dict(d):
    return {k: v[1] if isinstance(v, list) else v for k, v in d.items()}

# %%
# function to save the output of the simulation the way we want - gets called in the function above
def simOut(summary):
    # histogram of load factors in the following buckets 0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.85, 0.85-0.9, 0.9-0.95, 0.95-1
    lf = summary.legs
    # 0-0.5
    lf_0_5 = lf[lf['avg_load_factor'] <= 50]
    lf_0_5_count = len(lf_0_5)

    # 0.5-0.6
    lf_5_6 = lf[(lf['avg_load_factor'] > 50) & (lf['avg_load_factor'] <= 60)]
    lf_5_6_count = len(lf_5_6)

    # 0.6-0.7
    lf_6_7 = lf[(lf['avg_load_factor'] > 60) & (lf['avg_load_factor'] <= 70)]
    lf_6_7_count = len(lf_6_7)

    # 0.7-0.8
    lf_7_8 = lf[(lf['avg_load_factor'] > 70) & (lf['avg_load_factor'] <= 80)]
    lf_7_8_count = len(lf_7_8)

    # 0.8-0.85
    lf_8_85 = lf[(lf['avg_load_factor'] > 80) & (lf['avg_load_factor'] <= 85)]
    lf_8_85_count = len(lf_8_85)

    # 0.85-0.9
    lf_85_9 = lf[(lf['avg_load_factor'] > 85) & (lf['avg_load_factor'] <= 90)]
    lf_85_9_count = len(lf_85_9)

    # 0.9-0.95
    lf_9_95 = lf[(lf['avg_load_factor'] > 90) & (lf['avg_load_factor'] <= 95)]
    lf_9_95_count = len(lf_9_95)

    # 0.95-1
    lf_95_1 = lf[(lf['avg_load_factor'] > 95) & (lf['avg_load_factor'] <= 100)]
    lf_95_1_count = len(lf_95_1)

    # Create a dictionary to store the load factor distribution percentages
    lf_distribution_percentages = {
        '0-0.5': (lf_0_5_count / len(lf)),
        '0.5-0.6': (lf_5_6_count / len(lf)),
        '0.6-0.7': (lf_6_7_count / len(lf)),
        '0.7-0.8': (lf_7_8_count / len(lf)),
        '0.8-0.85': (lf_8_85_count / len(lf)),
        '0.85-0.9': (lf_85_9_count / len(lf)),
        '0.9-0.95': (lf_9_95_count / len(lf)),
        '0.95-1': (lf_95_1_count / len(lf))
    }

    # convert the dictionary to a dataframe
    lfd = pd.DataFrame(list(lf_distribution_percentages.items()), columns=['Load Factor Percentage', 'Percentage'])
    # Set 'Load Factor Percentage' as the index, then transpose the DataFrame
    lfd = lfd.set_index('Load Factor Percentage').transpose()


    # Rename the columns to have 'LF' as the prefix
    lfd.columns = ['LF_' + col for col in lfd.columns]
    lfd.reset_index(drop=True, inplace=True)

    carriers = summary.carriers
    # average of load factors for each carrier
    sys_lf = carriers['sys_lf'].mean()/100

    lfd['sys_LF'] = sys_lf

    # calculating the fare class mix
    fcms = summary._raw_fare_class_mix
    # Sum the sold values for each booking class
    sold_sum_by_booking_class = fcms.groupby('booking_class')['sold'].sum()

    # Calculate the total number of sold 
    total_sold = sold_sum_by_booking_class.sum()

    # Calculate the percentage of sold for each booking class
    sold_percentage_by_booking_class = (sold_sum_by_booking_class / total_sold)

    # convert series to a dataframe with colmns booking_class and sold_percentage 
    sold_percentage_by_booking_class = sold_percentage_by_booking_class.reset_index()
    sold_percentage_by_booking_class.columns = ['booking_class', 'sold_percentage']

    fcm = sold_percentage_by_booking_class.set_index('booking_class').transpose()

    fcm.reset_index(drop=True, inplace=True)

    total, connect, local = 0,0,0
    for idx, row in summary.paths.iterrows():
        if row.num_legs == 1:
            local += row.gt_sold
        else:
            connect += row.gt_sold

    total = local + connect

    # Calculate the percentage of local and connecting passengers
    local_percentage = (local / total) 
    merged_df = pd.concat([lfd, fcm], axis=1)
    merged_df['Local'] = local_percentage

    

    return merged_df


def get_bins(data, num_bins=20):
    """The goal is to have 'nice' sized bins with a boundary at zero.  This will almost always give
       a few more bins than requested, but it makes the graphs look good :-) """
    min_x = data.min()
    max_x = data.max()
    width = (max_x - min_x) / num_bins
    mults = [0.00001, 0.00002, 0.00005, 0.0001]
    while True:
        if mults[0] < width < mults[-1]:
            break
        mults = [m*10 for m in mults]
    
    # Now get the bin sizes
    if mults[0] < width < mults[1]:
        m = mults[0]
    elif mults[1] < width < mults[2]:
        m = mults[1]
    elif mults[2] < width < mults[3]:
        m = mults[2]

    min_x2 = m * floor(min_x / m)
    max_x2 = m * ceil(max_x / m)
    range2 = max_x2 - min_x2
    num_bins2 = int(range2 / m)

    bins = np.linspace(min_x2, max_x2, num_bins2+1)
    return bins


def win_loss_histogram(data, carrier, num_bins=10, strategy1='UEEE', strategy2='EEEE'):
    min_x = data.min()
    max_x = data.max()
    width = (max_x - min_x) / num_bins
    custom_bins = get_bins(data, num_bins)
    pos_data, neg_data = data[data >= 0], data[data < 0]

    total_count = len(data)
    green_percent = 100 * len(pos_data) / total_count
    red_percent = 100 * len(neg_data) / total_count
    print(green_percent, red_percent)
    plt.hist(pos_data, bins=custom_bins, color='green', edgecolor='black', label=f'Positive {strategy1}>{strategy2} - {green_percent:.2f}%')
    plt.hist(neg_data, bins=custom_bins, color='red', edgecolor='black', label=f'Negative {strategy1}<{strategy2} - {red_percent:.2f}%')
    
    # Calculate mean and standard deviation
    mean = data.mean()
    std_dev = data.std()

    # Plot vertical lines at mean, first standard deviation, and second standard deviation
    plt.axvline(x=mean, color='black', linestyle='--', label='Mean')
    plt.axvline(x=mean + std_dev, color='blue', linestyle='--', label='Mean ± 1 Std Dev')
    plt.axvline(x=mean - std_dev, color='blue', linestyle='--')
    plt.axvline(x=mean + 2 * std_dev, color='orange', linestyle='--', label='Mean ± 2 Std Dev')
    plt.axvline(x=mean - 2 * std_dev, color='orange', linestyle='--')

    # count the number of samples in each of the standard deviation ranges separately on either side of the mean - 0-1, 1-2
    count_0std = len(data[(data < mean - 2 * std_dev)])
    count_1std = len(data[(data >= mean - 2 * std_dev) & (data <= mean - std_dev)])
    count_2std = len(data[(data >= mean - std_dev) & (data <= mean)])
    count_3std = len(data[(data >= mean) & (data <= mean + std_dev)])
    count_4std = len(data[(data >= mean + std_dev) & (data <= mean + 2 * std_dev)])
    count_5std = len(data[(data > mean + 2 * std_dev)])
    
    print("Data less than -2 std dev: ", count_0std)
    print("Data between -2 and -1 std dev: ", count_1std)
    print("Data between -1 and mean: ", count_2std)
    print("Data between mean and +1 std dev: ", count_3std)
    print("Data between +1 and +2 std dev: ", count_4std) 
    print("Data greater than +2 std dev: ", count_5std) 
    
    plt.title(f' % Revenue Difference Histogram for {carrier}')
    plt.xlabel('Revenue % Difference')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import plot_importance

def plot_model_evaluation(model, x_test, y_test, feature_names=None, strategy1='UEEE', strategy2='EEEE'):
    """
    Generates and displays the following evaluation plots for the given model:
    1. Feature Importance
    2. Confusion Matrix
    3. ROC Curve and AUC
    
    Parameters:
        model: Trained model (e.g., XGBoost)
        x_test: Test features
        y_test: True labels for test set
        feature_names: List of feature names (optional, used for feature importance plot)
    
    Returns:
        None
    """
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=10, importance_type='weight', height=0.8)
    plt.title('Feature Importance', fontsize=16)
    plt.grid(False)

    ax = plt.gca()
    for bar in ax.patches:
        bar.set_color('skyblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Importance Score', fontsize=13)
    plt.ylabel('Features', fontsize=13)
    if feature_names is not None:
        ax.set_yticklabels([feature_names[int(idx.get_text())] for idx in ax.get_yticklabels()])
    plt.show()
    
    # 2. Confusion Matrix
    y_pred = model.predict(x_test)
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

    # 3. ROC Curve and AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = model.decision_function(x_test)  # For models without predict_proba
    
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

    # SHAP Summary Plot (Shows how each feature affects revenue strategy choice)
    explainer = shap.Explainer(model)
    shap_values = explainer(x_test)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, x_test, feature_names=feature_names)