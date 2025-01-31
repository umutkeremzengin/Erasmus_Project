from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from constants import *
import pandas as pd
import numpy as np
import os


def train_knn_model(df, feature_columns, label_column):
    # Ensure label_column is a list for consistency
    if not isinstance(label_column, list):
        label_column = [label_column]

    # Drop columns not in feature_columns or label_column
    columns_to_keep = feature_columns + label_column
    df_model = df[columns_to_keep].copy()

    # Split the DataFrame into features (X) and label (y)
    X = df_model[feature_columns]
    y = df_model[label_column[0]]  # Assuming label_column is a singleton list

    # Split dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = knn.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return knn


def train_decision_tree_model(df, feature_columns, label_column, encodings_dictionary):
    if not isinstance(label_column, list):
        label_column = [label_column]

    columns_to_keep = feature_columns + label_column
    df_model = df[columns_to_keep].copy()

    X = df_model[feature_columns]
    y = df_model[label_column[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Generate original class names from encoding dictionary
    # original_class_names = [name for name, index in sorted(encodings_dictionary[label_column[0]].items(), key=lambda item: item[1])]

    # Visualization: Full Decision Tree
    # plt.figure(figsize=(80, 40))
    # plot_tree(decision_tree, feature_names=feature_columns, filled=True, rounded=True, fontsize=10)
    # plt.title("Full Decision Tree Structure", fontsize=14)
    # full_tree_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'full_decision_tree_visual.png')
    # plt.savefig(full_tree_path, format='png', bbox_inches='tight', dpi=300)
    # plt.show()

    # Visualization: Decision Tree First 6 Levels
    plt.figure(figsize=(80, 40))
    plot_tree(decision_tree, feature_names=feature_columns, filled=True, rounded=True, fontsize=10, max_depth=6)
    plt.title("Decision Tree Structure (First 6 Levels)", fontsize=14)
    first_six_levels_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'decision_tree_first_6_levels.png')
    plt.savefig(first_six_levels_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()

    return decision_tree


def train_naive_bayes_model(data):
    # Split features and target
    X = data.drop('punctuality', axis=1)  # Features
    y = data['punctuality']  # Target

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Gaussian Naive Bayes classifier
    model = GaussianNB()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def train_random_forest_model(data):
    """
    Train a Random Forest model with the provided data.

    Args:
    - data (DataFrame): DataFrame containing the prepared data with features and target.

    Returns:
    - model: Trained Random Forest model.
    - y_pred_proba (DataFrame): Predicted probabilities for each class with class names.
    """
    # Split features and target
    X = data.drop('punctuality', axis=1)  # Features
    y = data['punctuality']  # Target

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Random Forest classifier
    model = RandomForestClassifier(n_estimators=30, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)

    # Map class labels to corresponding names
    class_names = {i: str(i) for i in model.classes_}
    class_names[-4] = '-4'
    class_names[-3] = '-3'
    class_names[-2] = '-2'
    class_names[-1] = '-1'

    y_pred_proba = y_pred_proba.rename(columns=class_names)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))

    # Print lengths
    print("Length of y_pred_proba:", len(y_pred_proba))
    print("Length of test data:", len(X_test))

    return model, y_pred_proba

