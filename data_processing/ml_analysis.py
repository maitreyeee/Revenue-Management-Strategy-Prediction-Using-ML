import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict, Any
import shap
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        
    Returns:
        Tuple containing the best model and its parameters
    """
    rf_model = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model and generate predictions.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target
        
    Returns:
        Tuple containing predictions and prediction probabilities
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_proba

def plot_model_results(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, 
                      model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate and save evaluation plots.
    
    Args:
        y_test (np.ndarray): Test target
        y_pred (np.ndarray): Model predictions
        y_pred_proba (np.ndarray): Prediction probabilities
        model: Trained model
        X (pd.DataFrame): Feature dataframe
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    plt.close()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    return feature_importance

def save_model_results(model: RandomForestClassifier, y_test: np.ndarray, y_pred: np.ndarray, 
                      feature_importance: pd.DataFrame, best_params: Dict[str, Any]) -> None:
    """
    Save model results and metrics to files.
    
    Args:
        model: Trained model
        y_test (np.ndarray): Test target
        y_pred (np.ndarray): Model predictions
        feature_importance (pd.DataFrame): Feature importance dataframe
        best_params (Dict[str, Any]): Best model parameters
    """
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    
    with open('results/model_metrics.txt', 'w') as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        
    Returns:
        Tuple containing the best model and its parameters
    """
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        
    Returns:
        Tuple containing the best model and its parameters
    """
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Define parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Perform grid search manually
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    current_params = params.copy()
                    current_params.update({
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree
                    })
                    
                    # Train model with current parameters
                    model = xgb.train(
                        current_params,
                        dtrain,
                        num_boost_round=100,
                        verbose_eval=False
                    )
                    
                    # Make predictions
                    preds = model.predict(dtrain)
                    preds = (preds > 0.5).astype(int)
                    
                    # Calculate accuracy
                    accuracy = (preds == y_train).mean()
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_params = current_params
                        best_model = model
    
    # Create scikit-learn compatible classifier with best parameters
    xgb_classifier = xgb.XGBClassifier(
        **best_params,
        n_estimators=100,
        use_label_encoder=False
    )
    xgb_classifier.fit(X_train, y_train)
    
    return xgb_classifier, best_params

def compare_models(models: Dict[str, Tuple[Any, Dict[str, Any]]], 
                  X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models (Dict): Dictionary of model names and their (model, params) tuples
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target
        
    Returns:
        pd.DataFrame: Comparison of model metrics
    """
    results = []
    
    for model_name, (model, params) in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'AUC': roc_auc,
            'Parameters': params
        })
    
    return pd.DataFrame(results)

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Plot comparison of model performance.
    
    Args:
        results_df (pd.DataFrame): Results from compare_models
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/model_accuracy_comparison.png')
    plt.close()
    
    # Plot AUC comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='AUC', data=results_df)
    plt.title('Model AUC Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/model_auc_comparison.png')
    plt.close() 