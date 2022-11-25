import talib
import datetime
import yfinance as yf
import holidays
import pandas as pd
from joblib import load

class FeatureGenerator():
    def __init__(self, ticker):
        self.ticker = ticker
        self.date = datetime.date.today()
        self.historical_data = self.download_historical_data()[0]
        self.TAs = self.get_TAs_for_all_windows()
        self.date_to_be_predicted = self.date
        self.date_related_features = self.gen_date_features()
        self.scaled_features = self.scale_features()
        self.current_close = self.download_historical_data()[1]

    def download_historical_data(self):
        df = yf.download(self.ticker, period='1mo')
        current_close = df['Close'][-1]
        return df, current_close
    
    def get_TAs(self, windowSize):
        """
        Function to get TA features
        """
        close_series = self.historical_data['Close'][-1*(windowSize+1):]
        high_series = self.historical_data['High'][-1*(windowSize+1):]
        low_series = self.historical_data['Low'][-1*(windowSize+1):]
        volume_series = self.historical_data['Volume'][-1*(windowSize+1):]

        ma = talib.MA(close_series, timeperiod=windowSize)[-1]
        rsi = talib.RSI(close_series, timeperiod=windowSize)[-1]
        mfi = talib.MFI(high_series, low_series, close_series, volume_series, timeperiod=windowSize)[-1]
        std = close_series.rolling(windowSize).std()[-1]

        return ma, rsi, mfi, std
        
    def get_TAs_for_all_windows(self):
        ma_dict = {}
        rsi_dict = {}
        mfi_dict = {}
        std_dict = {}
        for i in [7, 14, 21]:
            ma, rsi, mfi, std = self.get_TAs(i)
            ma_dict[f'{i} MA'] = ma
            rsi_dict[f'{i} RSI'] = rsi
            mfi_dict[f'{i} MFI'] = mfi
            if i == 7:
                std_dict[f'{i} STD DEV'] = std
        return ma_dict, rsi_dict, mfi_dict, std_dict
    
    def gen_date_features(self):
        """
        Generate date related features from date that 
        is wanted to be predicted

        We want to predict next business day's Close price
        """
        one_day = datetime.timedelta(days=1)
        next_day = self.date + one_day
        while next_day in holidays.WEEKEND:
            next_day += one_day

        self.date_to_be_predicted = next_day

        data = {'Date': [self.date_to_be_predicted]}
        df = pd.DataFrame(data)
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        data['dayofweek'] = df.index.dayofweek.values[0]
        data['quarter'] = df.index.quarter.values[0]
        data['month'] = df.index.month.values[0]
        data['year'] = df.index.year.values[0]
        data['dayofyear'] = df.index.dayofyear.values[0]
        data['dayofmonth'] = df.index.day.values[0]
        data['weekofyear'] = df.index.isocalendar().week.values[0]
        return data
    
    def scale_features(self):
        ticker = self.ticker[:4].lower()
        features_scaler = load(f'../experiments_final/feature_engineering/{ticker}_features_scaler.bin')
        TA_features = self.TAs
        date_features = self.date_related_features
        data = {
            '7 DAYS MA': [TA_features[0]['7 MA']],
            '14 DAYS MA': [TA_features[0]['14 MA']],
            '21 DAYS MA': [TA_features[0]['21 MA']],
            '7 DAYS STD DEV': [TA_features[3]['7 STD DEV']],
            'RSI 7': [TA_features[1]['7 RSI']],
            'RSI 14': [TA_features[1]['14 RSI']],
            'RSI 21': [TA_features[1]['21 RSI']],
            'MFI 7': [TA_features[2]['7 MFI']],
            'MFI 14': [TA_features[2]['14 MFI']],
            'MFI 21': [TA_features[2]['21 MFI']],
            'dayofweek': [date_features['dayofweek']],
            'quarter': [date_features['quarter']],
            'month': [date_features['month']],
            'year': [date_features['year']],
            'dayofyear': [date_features['dayofyear']],
            'dayofmonth': [date_features['dayofmonth']],
            'weekofyear': [date_features['weekofyear']],
        }
        
        df = pd.DataFrame(data)
        # features = [TA_features['7 MA'], TA_features['14 MA'], TA_features['21 MA'],
        #     TA_features['7 STD DEV'], 
        #     TA_features['7 RSI'], TA_features['14 RSI'], TA_features['21 RSI'],
        #     TA_features['7 MFI'], TA_features['14 MFI'], TA_features['21 MFI'],
        #     date_features['dayofweek'], date_features['quarter'],
        #     date_features['month'], date_features['year'], 
        #     date_features['dayofyear'], date_features['dayofmonth'],
        #     date_features['weekofyear']]
        scaled_features = features_scaler.transform(df)
        return scaled_features

    



