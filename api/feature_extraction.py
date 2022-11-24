"""
Module to extract features
"""
import talib

class Feature():
    pass

def generate_TAs_features(df):
    """
    Function to generate Technical Analysis features:
    - MA (Moving Average)
    - RSI (Relative Strength Index)
    - MFI (Money Flow Index)
    """
    res = df.copy()
    for i in [7, 14, 21]:
        res[f'{i} DAYS MA'] = talib.MA(res['Close'], timeperiod=i)
        res[f'{i} DAYS MA'] = res[f'{i} DAYS MA'].shift(1)
        res[f'RSI {i}'] = talib.RSI(res['Close'], timeperiod=i)
        res[f'RSI {i}'] = res[f'RSI {i}'].shift(1)
        res[f'MFI {i}'] = talib.MFI(res['High'], res['Low'], 
                                res['Close'], res['Volume'], 
                                timeperiod=i)
        res[f'MFI {i}'] = res[f'MFI {i}'].shift(1)

        if i == 7:
            res[f'{i} DAYS STD DEV'] = res['Close'].rolling(i).std()
            res[f'{i} DAYS STD DEV'] = res[f'{i} DAYS STD DEV'].shift(1)

    print(res.isnull().sum())
    res = res.dropna()
    return res

def generate_date_related_features(df):
    """
    Function to generate date related features
    """
    res = df.copy()
    res['dayofweek'] = df.index.dayofweek
    res['quarter'] = df.index.quarter
    res['month'] = df.index.month
    res['year'] = df.index.year
    res['dayofyear'] = df.index.dayofyear
    res['dayofmonth'] = df.index.day
    res['weekofyear'] = df.index.isocalendar().week
    return res