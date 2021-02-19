# This script allows for saving and loading forecasting results from models, including model, model name, scores, etc.

import os
import pickle

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
    filepath = os.getcwd()
    filepath = filepath[:len(filepath) - 18]
    print(filepath)
    # Final path should look like:
    # '{workspace}$src\\results\\validation or test\\from-date_to-date\\'
    # File should look like:
    # 'name_id'
    start_date = forecast.index[0].strftime('%Y-%m-%d')
    end_date = forecast.index[-1].strftime('%Y-%m-%d')
    filepath += '\\results\\' + folder + '\\' + start_date + '_' + end_date + '\\' + name + '_' + str(id_) + ".pkl"
    tuple_model = (id_, name, model, forecast, scores, folder)
    with open(filepath, 'wb') as file:
        pickle.dump(tuple_model, file)

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        tuple_model = pickle.load(file)

    return tuple_model
