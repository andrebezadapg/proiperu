
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import math

def plot_curves(data_dict:dict) -> None:
    '''
    Plot a response curve per media vehicle
    
    Args:
        data_dict: collection of curve's data

    Returns:
        A plot with all media response curves
    '''
    positions = range(1, len(data_dict.keys()) + 1)
    sizes = int(np.ceil(len(data_dict.keys())/2))

    fig = plt.figure(1)
    fig.suptitle('Nielsen Response Curves')
    fig.set_size_inches(18.5, 10.5)

    for i, k in enumerate(data_dict.keys()):
        ax = fig.add_subplot(sizes, sizes, positions[i])
        ax.plot(data_dict[k]['data']['Weekly Support'], data_dict[k]['data']['Prediction'])
        ax.title.set_text(k)
    plt.show()
    
def hill(x:float, a:float, b:float, c:float) -> float:
    '''
    Hill function
    
    Args:
        x: input value
        a, b,c: parameters to tune

    Returns:
        Use to estimated value for x, given parameters a, b and c
    '''
    return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))

def poly2(x:float, a:float, b:float, c:float) -> float:
    '''
    Polynomial of degree 2 function
        
    Args:
        x: input value
        a, b,c: parameters to tune

    Returns:
        Use to estimated value for x, given parameters a, b and c
    '''
    return a * x + b * x**2 + c

def poly3(x:float, a:float, b:float, c:float) -> float:
    '''
    Polynomial of degree 3 function
        
    Args:
        x: input value
        a, b,c: parameters to tune

    Returns:
        Use to estimated value for x, given parameters a, b and c
    '''
    return a * x + b * x**2 + c * x**3

def log_(x:float, a:float, b:float, c:float) -> float:
    '''
    Log function
        
    Args:
        x: input value
        a, b,c: parameters to tune

    Returns:
        The estimated value for x, given parameters a, b and c
    '''
    return a * np.log(b + x) + c

def cdf_(x:float, mu:float, sigma:float, c:float) -> float:
    '''
    CDF function
        
    Args:
        x: input value
        mu, sigma, c: parameters to tune

    Returns:
        The estimated value for x, given parameters a, b and c
    '''
    return scipy.stats.norm(mu,sigma).cdf(x) + c

def weibull(u:float, shape:float, scale:float) -> float:
    '''
    Weibull distribution
    
    Args:
        u: speed
        shape: shape parameter k
        scale: parameter A
        
    Returns:
        The estimated value for u, shape, and scale
    '''
    return sum(log(stats.weibull_min.pdf(x, shape, 0., u)))

def find_best_func(funcs:list, x:pd.Series, y:pd.Series, poly1d=True) -> tuple:
    '''
    Find the best function (r-squared-wise) to fit the original data per vehicle
    
    Args:
        funcs: list of functions to use use
        x: normalized impressions/trps
        y: response

    Returns:
        The best fitted function and its parameters
    '''
    best_func = None
    best_params = None
    max_fit = -np.inf

    for func in funcs:
        try:
            initialParameters = np.array([np.random.random(), np.random.random(), np.random.random()])
            fittedParameters, pcov = curve_fit(func, x, y, initialParameters, method='lm')
            modelPredictions = func(x, *fittedParameters) 
            absError = modelPredictions - y
            Rsquared = 1.0 - (np.var(absError) / np.var(y))
            
            if Rsquared > max_fit:
                best_func = func
                best_params = fittedParameters
                max_fit = Rsquared
        except:
            continue
    if poly1d:
        func = np.poly1d(np.polyfit(x, y, 5))
        fittedParameters = []
        modelPredictions = func(x)
        absError = modelPredictions - y
        Rsquared = 1.0 - (np.var(absError) / np.var(y))

        if Rsquared > max_fit:
            best_func = func
            best_params = fittedParameters
            max_fit = Rsquared
            
    print('r2:', round(max_fit,3))
    
    return best_func, best_params

def norm_data(data_dict:dict) -> dict:
    '''
    Min-max normalize x-axis of vehicle's data
    
    Args:
        data_dict: collection of curve's data

    Returns:
        A dictionary with data normalized
    '''
    for k in data_dict.keys():
        scaler = MinMaxScaler()
        df = data_dict[k]['data'].copy()
        df['Weekly Support'] = scaler.fit_transform(df[['Weekly Support']])
        data_dict[k]['data_norm'] = df
        data_dict[k]['scaler'] = scaler
    return data_dict

def fit_curves(funcs:list, data_dict:dict, poly1d:bool=True) -> dict:
    '''
    Fit all response curves and update dict
    
    Args:
        funcs: list of functions to use use
        data_dict: collection of curve's data

    Returns:
        A dictionary with fitted curves
    '''
    data_dict = norm_data(data_dict)

    for k in data_dict.keys():
        print(k)
        print('-'*10)
        tmp = data_dict[k]['data_norm'].copy()
        x = tmp['Weekly Support']
        y = tmp['Prediction']

        best_func, best_params = find_best_func(funcs, x, y, poly1d=poly1d)

        tmp['pred'] = tmp['Weekly Support'].apply(lambda x: best_func(x, *best_params))
        data_dict[k]['fitted'] = [best_func, best_params]
        print()
        
    return data_dict

def plot_fitted_curves(data_dict:dict) -> None:
    '''
    Plot a response curve per media vehicle and its fitted function
    
    Args:
        data_dict: collection of curve's data

    Returns:
        A plot with all media response curves and its corresponding fitted function
    
    '''
    positions = range(1, len(data_dict.keys()) + 1)
    sizes = np.ceil(len(data_dict.keys())/2)

    fig = plt.figure(1)
    fig.suptitle('Nielsen Response Curves')
    fig.set_size_inches(18.5, 10.5)

    for i, k in enumerate(data_dict.keys()):
        ax = fig.add_subplot(sizes, sizes, positions[i])
        ax.plot(data_dict[k]['data_norm']['Weekly Support'], data_dict[k]['data_norm']['Prediction'], 'r')
        ax.plot(data_dict[k]['data_norm']['Weekly Support'], data_dict[k]['data_norm']['Weekly Support'].apply(lambda x: data_dict[k]['fitted'][0](x, *data_dict[k]['fitted'][1])), 'b', linestyle='dashed')
        ax.title.set_text(k)
    plt.show()
    
def plot_actual_curves(data_dict:dict, x:np.ndarray) -> None:
    '''
    Plot a response curve per media vehicle with it's actual value
    
    Args:
        data_dict: collection of curve's data
        x: array with solution values

    Returns:
        A plot with all media response curves and their corresponding solutions
    
    '''
    positions = range(1, len(data_dict.keys()) + 1)
    sizes = np.ceil(len(data_dict.keys())/2)

    fig = plt.figure(1)
    fig.suptitle('Nielsen Response Curves')
    fig.set_size_inches(18.5, 10.5)

    for i, k in enumerate(data_dict.keys()):
        point = data_dict[k]['fitted'][0](x[i], *data_dict[k]['fitted'][1]) 
        ax = fig.add_subplot(sizes, sizes, positions[i])
        ax.plot(data_dict[k]['data_norm']['Weekly Support'], data_dict[k]['data_norm']['Prediction'])
        ax.plot(x[i], point, 'ro') 
        ax.title.set_text(k)
    plt.show()

def plot_solution_curves(data_dict:dict, x:np.ndarray, actual:np.ndarray) -> None:
    '''
    Plot a response curve per media vehicle with it's solution value
    
    Args:
        data_dict: collection of curve's data
        x: array with solution values
        actual: array with actual values

    Returns:
        A plot with all media response curves and their corresponding solutions
    
    '''
    positions = range(1, len(data_dict.keys()) + 1)
    sizes = int(np.ceil(len(data_dict.keys())/2))

    fig = plt.figure(1)
    fig.suptitle('Nielsen Response Curves')
    fig.set_size_inches(18.5, 10.5)

    for i, k in enumerate(data_dict.keys()):
        point = data_dict[k]['fitted'][0](x[i], *data_dict[k]['fitted'][1])
        point_actual = data_dict[k]['fitted'][0](actual[i], *data_dict[k]['fitted'][1]) 
        ax = fig.add_subplot(sizes, sizes, positions[i])
        ax.plot(data_dict[k]['data_norm']['Weekly Support'], data_dict[k]['data_norm']['Prediction'])
        ax.plot(x[i], point, 'go')
        ax.plot(actual[i], point_actual, 'ro') 
        ax.title.set_text(k)
    plt.show()

### Optimization
def get_spend(s:float, data_dict:dict, vehicle:str) -> float:
    '''
    Compute spend given impressions: imp * cpp / ix * w
    
    Args:
        s: normed impressions
        data_dict: collection of curve's data
        vehicle: media vehicle to compute for

    Returns:
        Spend for those impressions on that vehicle
    '''
    return data_dict[vehicle]['scaler'].inverse_transform([[s]])[0][0]  * data_dict[vehicle]['cpp'] / data_dict[vehicle]['ix_spend'] * data_dict[vehicle]['weeks'] 

def get_nos(y:float, data_dict:dict, vehicle:str) -> float:
    '''
    Compute NOS given impressions: imp / ix * w * su
    
    Args:
        y: normed impressions
        data_dict: collection of curve's data
        vehicle: media vehicle to compute for
        
    Returns:
        NOS for those impressions on that vehicle
    '''
    return data_dict[vehicle]['scaler'].inverse_transform([[y]])[0][0] / data_dict[vehicle]['ix_nos'] * data_dict[vehicle]['weeks'] * data_dict[vehicle]['fitted'][0](y, *data_dict[vehicle]['fitted'][1])
