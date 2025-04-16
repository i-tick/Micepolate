import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def display_missingness_heatmap():
    # Check if the necessary file is available in session state
    if "file" not in st.session_state:
        st.error("Please upload the dataset first.")
        return
    
    # Access the dataframe from session state
    target_df = st.session_state["file"]

    # Ensure that the DataFrame has numerical columns for regression
    all_columns = target_df.columns.tolist()
    numerical_columns = target_df.select_dtypes(include=[float, int]).columns.tolist()

    if len(numerical_columns) < 2:
        st.error("Not enough numerical columns to run the logistic regression.")
        return

    # Create an empty matrix for storing the coefficients
    coef_matrix = pd.DataFrame(np.zeros((len(numerical_columns), len(numerical_columns))),
                               columns=numerical_columns, index=numerical_columns)

    # Loop through each column and use logistic regression to predict missingness
    for target_col in all_columns:
        # Create a mask for missing values (True for NaN, False for non-NaN)
        target_nan_mask = target_df[target_col].isna()

        # Prepare predictor variables (all other columns except the target column)
        predictors = target_df.drop(columns=[target_col]).select_dtypes(include=[float, int]).copy()
        predictors = predictors.fillna(predictors.mean())

        # Standardize the predictors for logistic regression
        scaler = StandardScaler()
        predictors_scaled = scaler.fit_transform(predictors)

        # Initialize logistic regression model
        model = LogisticRegression(max_iter=1000, solver='lbfgs')

        # Split data into train and test sets for logistic regression
        X_train, X_test, y_train, y_test = train_test_split(predictors_scaled, target_nan_mask, test_size=0.2, random_state=42)

        try:
            # Fit the logistic regression model
            model.fit(X_train, y_train)

            # Get the coefficients for the predictors
            coef = model.coef_[0]

            # Populate the matrix with coefficients
            for i, predictor in enumerate(predictors.columns):
                coef_matrix.at[target_col, predictor] = coef[i]
        except:
            for i, predictor in enumerate(predictors.columns):
                coef_matrix.at[target_col, predictor] = 0

    coef_matrix = coef_matrix.T

    min_val = coef_matrix.min().min()  # Minimum value in the DataFrame
    max_val = coef_matrix.max().max()  # Maximum value in the DataFrame
    max_jitter = max([-min_val, max_val])
    min_val, max_val = (-max_jitter, max_jitter)

    # Scale the DataFrame
    coef_matrix = 2 * (coef_matrix - min_val) / (max_val - min_val) - 1

    # Plot the heatmap of the coefficient matrix
    plt.figure(figsize=(8, 4))
    sns.heatmap(coef_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", cbar=True)
    plt.title("Significance of Columns in Predicting Missingness")
    plt.ylabel("Predictor Columns")
    plt.xlabel("Target Columns")
    plt.tight_layout()
    st.pyplot(plt)

# Entry point to trigger the heatmap
def populateMissingnessSignificanceHeatmap():
    display_missingness_heatmap()
