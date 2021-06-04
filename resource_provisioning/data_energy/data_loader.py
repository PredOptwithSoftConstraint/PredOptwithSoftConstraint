import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore

# NERC6 holidays with inconsistent dates. Created with python holidays package
# years 1990 - 2024
with open('data_energy/holidays.pickle', 'rb') as f:
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


def make_features(region, noise=2.5, hours_prior=24):
    if region not in REGIONS:
        raise ValueError("Invalid region {region}".format(region=region))

    df = pd.read_csv(region + '.csv', parse_dates={'dates':['year','month','day']}, 
                     infer_datetime_format=True)
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

    return r_df


def transform_data(data, timesteps, var='x'):
  m = []
  s = data.to_numpy()
  for i in range(s.shape[0]-timesteps):
      m.append(s[i:i+timesteps].tolist())

  if var == 'x':
      t = np.zeros((len(m), len(m[0]), len(m[0][0])))
      for i, x in enumerate(m):
          for j, y in enumerate(x):
              for k, z in enumerate(y):
                  t[i, j, k] = z
  else:
      t = np.zeros((len(m), len(m[0])))
      for i, x in enumerate(m):
          for j, y in enumerate(x):
              t[i, j] = y
  return t


def transform_data(df, timesteps):
    # 2: make targets
    m = []
    s = df['load'].to_numpy()
    for i in range(s.shape[0]-24):
        m.append(s[i:i+24].tolist())

    t = np.zeros((len(m), len(m[0]), len(m[0][0])))
    for i, x in enumerate(m):
        for j, y in enumerate(x):
            for k, z in enumerate(y):
                t[i, j, k] = z

    d_df = pd.DataFrame(data=t, columns=["d"+str(i) for i in range(24)])
    return r_df, d_df 
