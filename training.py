"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""


import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import joblib
from submission import clean_df # for clean_df function


def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    X = model_df.drop(['new_child', 'nomem_encr'], axis=1)
    y = model_df['new_child']

    # Define the model
    model = LogisticRegression(verbose=10)

    # Fit the model
    model.fit(X,y)

    # Save the model
    joblib.dump(model, "model.joblib")

    # Print progress
    print("Model saved.")


# Load training data
df_train = pd.read_csv("data/training_data/PreFer_train_data.csv", low_memory=False)
df_train_outcome = pd.read_csv("data/training_data/PreFer_train_outcome.csv")

# Clean training data using clean_df from submission.py
df_train_cleaned = clean_df(df_train)

# Train model and save
train_save_model(df_train_cleaned, df_train_outcome)