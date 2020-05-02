# %% [markdown]
# This is a simple notebook for H2O AutoMl prediction.
# MLflow used as tracking tool because experiments take long time complete
# and it is hard to manage too many experiments. 
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
h2o.init(max_mem_size='16G')

# Initialize mlflow experiment
mlflow.set_tracking_uri(f'..{os.sep}mlruns')
experiment_name = 'automl-benchmark'
try:
    experiment = mlflow.create_experiment(experiment_name)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

# Reading seeds
seed_path = f'..{os.sep}data{os.sep}seeds.txt'
seeds = []
with open(seed_path,mode ='r') as file:
    for seed in file:
        seed.strip(r'/n')
        seeds.append(int(seed))

dataset_name = 'ieee_fraud_detection'
data = pd.read_pickle(f'..{os.sep}data{os.sep}{dataset_name}{os.sep}{dataset_name}.pkl')

#%%
run_time_secs = 600
target_column = 'isFraud'
for seed in seeds:
    with mlflow.start_run(run_name='h2o'):
        # Split data into two parts (train, valid)
        train, valid = train_test_split(data, random_state = seed)
        train_df = h2o.H2OFrame(train)
        # preparing column data for validation frame it must have same types and names with train frame
        test_column_names = train_df.columns
        test_column_types = train_df.types
        valid_df = h2o.H2OFrame(valid, column_names=test_column_names, column_types=test_column_types)
        # Identify predictors and response
        x = train_df.columns
        x.remove(target_column)
        # For binary classification, response should be a factor
        train_df[target_column] = train_df[target_column].asfactor()
        valid_df[target_column] = valid_df[target_column].asfactor()
        # Run AutoML for 10 mins with AUC metric
        aml = H2OAutoML(max_runtime_secs = run_time_secs, 
                        seed=seed, 
                        stopping_metric='AUC',
                        sort_metric = 'AUC')
        aml.train(x=x, y=target_column, training_frame=train_df, validation_frame=valid_df)
        
        mlflow.log_metric('AUC', aml.leader.auc(valid=True))
        mlflow.log_param('seed', seed)
        mlflow.log_param('run_time', run_time_secs)
        mlflow.log_param('dataset_name', dataset_name)
        mlflow.log_param('model_name',aml.leaderboard[0,0])
        mlflow.h2o.log_model(aml.leader, 'model')