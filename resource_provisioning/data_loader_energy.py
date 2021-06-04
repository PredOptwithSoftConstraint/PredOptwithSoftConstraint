import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore

PATH = './data_energy/'

# NERC6 holidays with inconsistent dates. Created with python holidays package
# years 1990 - 2024
with open(PATH + 'holidays.pickle', 'rb') as f:
    nerc6 = pickle.load(f)

REGIONS = ['COAST', 'NCENT', 'NORTH', 'SCENT', 'SOUTH', 'WEST', 'EAST', 'FWEST']
HOLIDAYS = ["New Year's Day", "Memorial Day", "Independence Day", "Labor Day", 
            "Thanksgiving", "Christmas Day"]

def isHoliday(holiday, df):
    # New years, memorial, independence, labor day, Thanksgiving, Christmas
    m1 = None
    if holiday == "New Year's Day":
        m1 = (df["dates"].dt.month == 1) & (df["dates"].dt.day == 1)
    if holiday == "Independence Day":
        m1 = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 4)
    if holiday == "Christmas Day":
        m1 = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 25)
    m1 = df["dates"].dt.date.isin(nerc6[holiday]) if m1 is None else m1
    m2 = df["dates"].dt.date.isin(nerc6.get(holiday + " (Observed)", []))
    return m1 | m2


def add_noise(m, std):
    noise = np.random.normal(0, std, m.shape[0])
    return m + noise


def make_dataset(region, N, noise=2.5, hours_prior=24, fmt='day_hour'):
    """ Make dataset for one given region

        fmt: ['hour', 'day_hour']
        Return:
             X: ndarray of dimensions (day, hour, features) for fmt 'day_hour', 
                and (hour, features) for fmt 'hour'.
             y: ndarray of dimensions (day, hour) for fmt 'day_hour', 
                and (hour) for fmt 'hour'.
    """
    def _is_X_cached(region, fmt):
        return os.path.exists(PATH + region + '_X_cache_' + fmt + '.npy') 
    def _is_y_cached(region, fmt):
        return os.path.exists(PATH + region + '_y_cache_' + fmt + '.npy') 
    def _read_X_cache(region, fmt):
        X = np.load(PATH + region + '_X_cache_' + fmt + '.npy')
        return X
    def _read_y_cache(region, fmt):
        y = np.load(PATH + region + '_y_cache_' + fmt + '.npy')
        return y
    def _to_X_cache(ndarr, region, fmt):
        np.save(PATH + region + '_X_cache_' + fmt + '.npy', ndarr) 
    def _to_y_cache(ndarr, region, fmt):
        np.save(PATH + region + '_y_cache_' + fmt + '.npy', ndarr) 

    if region not in REGIONS:
        raise ValueError("Invalid region {region}".format(region=region))

    if fmt not in ['hour', 'day_hour']:
        raise ValueError("Invalid fmt {fmt}".format(fmt=fmt))

    if _is_X_cached(region, fmt) and _is_y_cached(region, fmt):
        X = _read_X_cache(region, fmt)
        y = _read_y_cache(region, fmt)
        return X, y

    df = pd.read_csv(PATH + region + '.csv', parse_dates={'dates':['year','month','day']}, 
                     infer_datetime_format=True, skiprows=lambda x: x > 0 and x < 148920 - N)
    df['dates'] = pd.to_datetime(df['dates'], format="%Y %m %d") + pd.to_timedelta(df['hour'], unit='h')
    df = df.drop('hour', axis=1)
    
    # 1: make the features
    r_df = pd.DataFrame()

    # LOAD
    r_df["load_n"] = zscore(df["load"])
    r_df["load_prev_n"] = r_df["load_n"].shift(hours_prior)
    r_df["load_prev_n"].fillna(method='bfill', inplace=True)
    
    # LOAD PREV
    def _chunks(l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]
    n = np.array([val for val in _chunks(list(r_df["load_n"]), 24) for _ in range(24)])
    l = ["l" + str(i) for i in range(24)]
    for i, s in enumerate(l):
        r_df[s] = n[:, i]
        r_df[s] = r_df[s].shift(hours_prior)
        r_df[s] = r_df[s].bfill()
    r_df.drop(['load_n'], axis=1, inplace=True)
    
    # date
    r_df["years_n"] = zscore(df["dates"].dt.year)
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.hour, prefix='hour')], axis=1)
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.dayofweek, prefix='day')], axis=1)
    r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.month, prefix='month')], axis=1)
    for holiday in HOLIDAYS:
        r_df[holiday] = isHoliday(holiday, df)

    # temperatures
    temp_noise = df['tempc'] + np.random.normal(0, noise, df.shape[0])
    r_df["temp_n"] = zscore(temp_noise)
    r_df['temp_n^2'] = zscore([x*x for x in temp_noise])

    if fmt == 'day_hour':
        X = data_transform(r_df, hours_prior, 'X')
        y = data_transform(df['load'], hours_prior, 'y')
    else:
        X = r_df.to_numpy()
        y = df['load'].to_numpy()

    _to_X_cache(X, region, fmt)
    _to_y_cache(y, region, fmt)
    return X, y


def data_transform(data, timesteps, var='X'):
    m = []
    s = data.to_numpy()
    for i in range(s.shape[0]-timesteps):
        m.append(s[i:i+timesteps].tolist())

    if var == 'X':
        t = np.zeros((len(m), len(m[0]), len(m[0][0])))
        for i, x in enumerate(m):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    t[i, j, k] = z
    elif var == 'y':
        t = np.zeros((len(m), len(m[0])))
        for i, x in enumerate(m):
            for j, y in enumerate(x):
                t[i, j] = y
    else:
        raise ValueError("Wrong var {var}, should be X or y".format(var))
    return t

