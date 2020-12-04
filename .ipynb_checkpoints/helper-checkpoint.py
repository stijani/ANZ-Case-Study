import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np


def create_lag_features(data, target_col):
    '''
    Create lag features from a time series data
    Parameters:
        data(dataframe): input data dataframe
        target_col(string): column in dataframe holding the series
        
    Return: dataframe with lag columns
    '''
    data['Last_1_Week_Flu_Counts'] = data[target_col].shift(1)
    data['Last_1_Week_difference'] = data['Last_1_Week_Flu_Counts'].diff()
    data['Last_2_Week_Flu_Counts'] = data[target_col].shift(2)
    data['Last_2_Week_difference'] = data['Last_2_Week_Flu_Counts'].diff()
    if 'Week Ending (Friday)' in list(data.columns):
        data.drop('Week Ending (Friday)', axis=1, inplace=True)
    data.dropna(inplace=True)
    return data


def plot_target_vs_pred(target, pred, weeks, legend_list):
    plt.plot(weeks, target)
    plt.plot(weeks, pred)
    plt.legend([legend_list[0], legend_list[1]])
    plt.xlabel('Week Number')
    plt.ylabel('Flu Count - Nation Wide')
    plt.title('Comparing Predicted Values to Target')
    
    
def rmsle(ytrue, ypred):
    return np.sqrt(mean_absolute_error(ytrue, ypred))


def make_final_model():
    model = XGBRegressor(objective='reg:squarederror', 
                               n_estimators=1000, 
                               learning_rate =0.01, 
                               max_depth=5, 
                               subsample=0.8)
    return model


def roll_forward_validation(model, data_df, split):
    '''Trains and runs validation on data in the roll forward format
    
        Parameters: 
            model(model object): model to train
            data_df(data frame): a data frame containing the training and validation data
            split(int): interger indicating the start index for validation data
        return:
            target_val(list): validation target
            predicted_val(list): list of prediction for the validation set
    '''
    mean_error = []
    predicted_val = []
    target_val = []
    weeks = []
    target_var = list(data_df.columns)[0]
    for week in range(split, len(data_df)):
        weeks.append(week)
        train = data_df.iloc[:week, :]
        val = data_df.iloc[[week]]
        target_val.append(val[target_var]) # append the val target

        X_train = train.drop(target_var, axis=1) 
        X_val = val.drop(target_var, axis=1)
        y_train, y_val = train[target_var].values, val[target_var].values

        model.fit(X_train, y_train) 
        pred = model.predict(X_val) 
        predicted_val.append(pred)
    error = mean_absolute_error(predicted_val, target_val)
    print('MAE Error: {}'.format(error))
    return target_val, predicted_val, weeks 