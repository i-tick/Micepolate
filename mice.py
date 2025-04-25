import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pymc as pm
import pymc_bart as pmb
import numpy as np


def implement_MICE(file, cols, max_iterator=25, rand_state=0):
    df_copy = file.copy()
    imputer = IterativeImputer(max_iter=max_iterator, random_state=rand_state)

    # Create a list to store values after each iteration
    iteration_values = []

    # Get all numerical columns
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    print('Numerical columns:', numerical_cols)

    # Fit the imputer and transform the data
    imputed_values = imputer.fit_transform(df_copy[numerical_cols])

    # Store the values after each iteration
    for i in range(max_iterator):
        # Get the current state of the imputer
        current_values = imputer.transform(df_copy[numerical_cols])
        iteration_values.append(current_values)

    # Extract only the imputed values for the specified columns
    imputed_cols_values = imputed_values[:, [numerical_cols.index(col) for col in cols]]
    print('Imputed values:', imputed_cols_values)
    
    # Update the DataFrame with the imputed values for the specified columns
    for i, col in enumerate(cols):
        df_copy[col] = imputed_cols_values[:, i]

    # print('Imputed',imputed_values)
    # print('Iteration',iteration_values)
    return df_copy, imputed_values




# def implement_MICE(file, cols_to_impute, cols_to_condition, max_iterator=25, rand_state=0):
#     df_copy = file.copy()
#     imputer = IterativeImputer(max_iter=max_iterator, random_state=rand_state)

#     # Combine the columns to condition and the columns to impute for imputation
#     combined_cols = cols_to_condition + cols_to_impute

#     # Fit the imputer on the combined columns and transform the data
#     imputed_values = imputer.fit_transform(df_copy[combined_cols])

#     # Update only the columns that need to be imputed
#     df_copy.loc[:, cols_to_impute] = imputed_values[:, len(cols_to_condition):]

#     return df_copy, imputed_values


def implement_BART(file, cols, rand_state=5781):
    df_copy = file.copy()
    RANDOM_SEED = rand_state
    # Separate columns with null values and those without
    null_cols = [col for col in cols if df_copy[col].isnull().any()]
    x = [col for col in df_copy.select_dtypes(include=[np.number]).columns if col not in null_cols and col not in cols]
    print('Columns without null values:', x)

    X_full = df_copy[x].copy()
    for col in cols:
        Y_full = df_copy[col].copy()

        missing_indices = Y_full[Y_full.isnull()].index
        print(len(missing_indices))

        # Split observed and missing
        observed_mask = Y_full.notnull()
        X_observed = X_full[observed_mask]
        Y_observed = Y_full[observed_mask]
        X_missing = X_full[~observed_mask]


        # Define MutableData outside the model for future updates
        with pm.Model() as model_impute:
            X_data = pm.Data("X_data", X_observed)
            α = pm.Exponential("α", 1)
            μ = pmb.BART("μ", X_data, np.log(Y_observed + 1), m=50)
            y = pm.NegativeBinomial("y", mu=pm.math.exp(μ), alpha=α, observed=Y_observed)
            idata_impute = pm.sample(random_seed=RANDOM_SEED, target_accept=0.95)

        # Use posterior to predict missing targets
        with model_impute:
            pm.set_data({"X_data": X_missing})
            μ_post = pm.sample_posterior_predictive(idata_impute, var_names=["μ"], random_seed=RANDOM_SEED)

        # Use the median or mean of predicted values
        y_pred  = np.exp(μ_post.posterior_predictive["μ"]).mean(axis=(0, 1)) - 1
        Y_full_imputed = Y_full.copy()
        Y_full_imputed[~observed_mask] = y_pred

        # Display before vs after
        print("Missing values before:", Y_full.isna().sum())
        print("Missing values after:", Y_full_imputed.isna().sum())
        df_copy[col] = Y_full_imputed




    return df_copy, Y_full_imputed
