# %% [markdown]
# This note
#%%
# Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o

# %%
# Initialize h2o server
h2o.init(max_mem_size='8G')

experiment_name = 'automl-benchmark'
try:
    experiment = mlflow.create_experiment(experiment_name)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri(f'..{os.sep}/mlruns')

dataset_name = 'home_credit_Default_risk'
data = pd.read_pickle(f'..{os.sep}data{os.sep}{dataset_name}{os.sep}{dataset_name}.pkl')

#%%
run_time_secs = 1200 
with mlflow.start_run():
    train_df = h2o.H2OFrame(data)
    # Identify predictors and response
    x = train_df.columns
    y = "TARGET"
    x.remove(y)
    # For binary classification, response should be a factor
    train_df[y] = train_df[y].asfactor()
    # Run AutoML for 10 mins with AUC metric
    aml = H2OAutoML(max_runtime_secs = run_time_secs, 
                    seed=1, 
                    stopping_metric='AUC',
                    sort_metric = 'AUC')
    aml.train(x=x, y=y, training_frame=train_df)
    mlflow.log_metric("AUC", aml.leader.auc())
    mlflow.log_param('seed', 42)
    mlflow.log_param('run_time', run_time_secs)
    mlflow.log_param('dataset_name', dataset_name)
    mlflow.log_param('model_name',aml.leaderboard[0,0])
    mlflow.h2o.log_model(aml.leader, "model")