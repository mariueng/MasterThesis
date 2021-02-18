# This script allows for saving and loading forecasting results from models, including model, model name, scores, etc.

import os
import pickle
import shutil

"""
Format that models are saved in:
tuple(id, name, model, forecast, scores, folder)

Descriptions:
    - id: Basic ID for model
    - name: Model name
    - model: Actual model, should work for any model type/object
    - forecast: The forecasted data as dataframe, includes point and intervals
    - scores: Calculated scores for trained and validated/tested model
    - folder: specifies whether the model was run on validation (trained model) or test data (tested model) and it is
    then put in the corresponding folder.
"""


def save_pickle(id_, name, model, forecast, scores, folder):
    #filepath = os.getcwd()
    #filepath = filepath[:filepath.index("src")]
    start_date = forecast.index[0].strftime('%Y-%m-%d')
    end_date = forecast.index[-1].strftime('%Y-%m-%d')
    filepath = get_result_folder(model) + '\\' + str(id_) + '_' + start_date + '_' + end_date + '.pkl'
    tuple_model = (id_, name, model, forecast, scores, folder)
    with open(filepath, 'wb') as file:
        pickle.dump(tuple_model, file)


def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        tuple_model = pickle.load(file)

    return tuple_model


def get_result_folder(model):
    creation_time = ''  # To avoid overwriting, change to: '_' + model.get_time()
    folder_path = '../results/validation' + (model.get_name() + creation_time).replace(' ', '')
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return folder_path
