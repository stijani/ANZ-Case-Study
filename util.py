import pandas as pd
import matplotlib.pyplot as plt
import datetime

def read_data_to_pandas(path, header_row=1):
    '''
    Reads data from excel and returns pandas dataframe
        Parameters:
                header_row (int): row in data to be used as header
                path (string): path to data
        Returns:
                data (pandas dataframe)
    '''
    data_raw = pd.read_excel(path, header=None)
    header = list(data_raw.loc[header_row, :])
    data = data_raw.drop([0, header_row])
    data.columns = header
    data.reset_index(drop=True, inplace=True)
    return data


def group_count_unstack(df, var1, var2, var3):
    '''
    Groups a dataframe by var1 and var2 in order. Then
    unstack the resulting series into a dataframe using var3
    
    Parameters:
                df (dataframe): input dataframe
                var1 (str): first variable to groupby
                var2 (str): second variable to groupby
                var3 (str): variable to unstack with
        Returns:
                pandas dataframe
    '''
    temp_df = df.copy()
    return temp_df.groupby([var1, var2])[var1].size().unstack(var3, fill_value=0)


def customized_plot(df, title='', xlabel='', ylabel='', fontsize=10, plot_size=(8, 6), loc=0):
    '''
    Plot all vriables in a dataframe
        Parameters:
                df (dataframe): a dataframe object
                title (string): title of the plot
                xlabel (string): x-axis label
                ylabel (string): y-axis label
                fontsize (int): label font size
                plot_size(tuple): figure size
                loc(int): legend placement
    '''
    plt.figure(figsize=plot_size)
    for col in df.columns:
        plt.plot(df[col], label=col)
    plt. xlabel(xlabel, fontsize=fontsize)
    plt. ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend()
    plt.show()
    return None

def strip_date(date, item='W'):
    '''
    Takes a date string and returns one of its components number
        parameters:
            date(string): a date in '%Y-%m-%d' format
            item(string): the component needed, default is week
            
        return: a string corresponding to the requested component
    '''
    date = date.split(' ')[0] #only date
    d = datetime.datetime.strptime(date,'%Y-%m-%d')
    return int(datetime.datetime.strftime(d,'%'+item))