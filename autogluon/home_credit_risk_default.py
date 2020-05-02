# %% [markdown]
# This is a simple notebook for Autogluon AutoMl prediction.
# MLflow used as tracking tool since experiments take long time complete
# and it is hard to manage too many experiments. 
#%%
# Importing necessary libraries
import os
import re
import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autogluon import TabularPrediction as task
import mlflow
import mlflow.gluon

# %%
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

dataset_name = 'home_credit_default_risk'
data = pd.read_pickle(f'..{os.sep}data{os.sep}{dataset_name}{os.sep}{dataset_name}.pkl')
# Renaming all the characters except for regex experresion
# For some reason lightgbm gives error with some column names
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

#%%
run_time_secs = 600
target_column = 'TARGET'
# Pickling other models require >1GB amount of space
# Used hyperparameters option to discard other models
hyper_parameters = {'NN':{},'GBM':{},'CAT':{},'LR':{} }
for seed in seeds:
    with mlflow.start_run(run_name='autogluon'):
        # Create output directory for auto gluon
        models_dir = 'AutogluonModels'
        random_dir = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = 12))
        output_dir = f'{models_dir}{os.sep}{random_dir}'
        os.mkdir(output_dir)
        # Split data into two parts (train, valid)
        train, valid = train_test_split(data, random_state = seed)
        predictor = task.fit(train_data=train, 
                            label=target_column,
                            problem_type = 'binary',
                            eval_metric = 'roc_auc', 
                            stopping_metric='roc_auc',
                            hyperparameters= hyper_parameters,
                            stack_ensemble_levels=2, 
                            time_limits = run_time_secs,
                            cache_data=False, 
                            verbosity = 2,
                            output_directory=output_dir)
        test_data = valid
        y_test = test_data[target_column]  # values to predict
        test_data_nolab = test_data.drop(labels=[target_column],axis=1) # delete label column to prove we're not cheating
        # AutoGluon will gauge predictive performance using 
        # evaluation metric: roc_auc this metric expects predicted probabilities 
        # rather than predicted class labels, so you'll need to use predict_proba() 
        # instead of predict()
        y_pred = predictor.predict_proba(test_data_nolab)
        score = roc_auc_score(y_test,y_pred)
        mlflow.log_metric('AUC', score)
        mlflow.log_param('seed', seed)
        mlflow.log_param('run_time', run_time_secs)
        mlflow.log_param('dataset_name', dataset_name)
        mlflow.log_param('model_name',predictor.leaderboard().iloc[0,0])
        mlflow.log_artifact(output_dir)