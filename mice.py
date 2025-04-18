import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def implement_MICE(file, cols, max_iterator=25, rand_state=0):
    df_copy = file.copy()
    imputer = IterativeImputer(max_iter=max_iterator, random_state=rand_state)

    # Create a list to store values after each iteration
    iteration_values = []

    # Fit the imputer and transform the data
    imputed_values = imputer.fit_transform(df_copy[cols])

    # Store the values after each iteration
    for i in range(max_iterator):
        # Get the current state of the imputer
        current_values = imputer.transform(df_copy[cols])
        iteration_values.append(current_values)

        # Print the values after each iteration
        # print(f"\nIteration {i + 1}:")
        # print(pd.DataFrame(current_values, columns=cols))

    df_copy.loc[:, cols] = imputed_values
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
