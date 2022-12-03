from flask import Flask, request
import pickle
import xgboost as xgb
from generate_features import FeatureGenerator
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route('/predict-next-day/<company>', methods=['GET'])
def predict_next_day(company):
    model = xgb.XGBRegressor()
    model.load_model(f'../experiments_final/XGBReg/models/{company}.json')
    
    # Generate Features
    ticker = company.upper()+'.JK'
    fg = FeatureGenerator(ticker=ticker)
    scaled_features = fg.scaled_features

    # Predict
    prediction = model.predict(scaled_features)

    # Inverse predicted
    close_scaler = load(f'../experiments_final/feature_engineering/{company}_close_scaler.bin')
    inversed = close_scaler.inverse_transform(np.array(prediction).reshape(-1,1))
    formatted_price = float("{:.2f}".format(inversed.tolist()[0][0]))
    return_val = {'data': {
            'price': str(formatted_price),
            'prediction_date': str(fg.date_to_be_predicted)
            }
        }
    return return_val

@app.route('/predict-next-day/all', methods=['GET'])
def predict_next_day_all():
    companies = ['tlkm', 'isat', 'fren', 'excl']
    models = {}
    scaled_features = {}
    predictions = {}
    inversed = {}
    return_pcts = {}

    for company in companies:
        ticker = company[:4].upper()+'.JK'

        # load models
        model = xgb.XGBRegressor()
        model.load_model(f'../experiments_final/XGBReg/models/{company}.json')
        models[company] = model

        # generate features
        fg = FeatureGenerator(ticker=ticker)
        print(ticker, fg.current_close)
        scaled_feats = fg.scaled_features
        scaled_features[company] = scaled_feats, fg.date_to_be_predicted

        # predictions
        pred = model.predict(scaled_feats)
        predictions[company] = pred

        # inversed
        close_scaler = load(f'../experiments_final/feature_engineering/{company}_close_scaler.bin')
        inv = close_scaler.inverse_transform(np.array(pred).reshape(-1,1))
        formatted_inv = float("{:.2f}".format(inv.tolist()[0][0]))
        return_pct = ((formatted_inv - fg.current_close) * 100) / fg.current_close
        return_pct = float("{:.2f}".format(return_pct))
        inversed[company] = formatted_inv
        return_pcts[company] = return_pct

    return_val = {
        'tlkm': {
            'price': str(inversed['tlkm']),
            'prediction_date': str(scaled_features['tlkm'][1]),
            'return_pct': str(return_pcts['tlkm'])
        },
        'isat': {
            'price': str(inversed['isat']),
            'prediction_date': str(scaled_features['isat'][1]),
            'return_pct': str(return_pcts['isat'])
        },
        'fren': {
            'price': str(inversed['fren']),
            'prediction_date': str(scaled_features['fren'][1]),
            'return_pct': str(return_pcts['fren'])
        },
        'excl': {
            'price': str(inversed['excl']),
            'prediction_date': str(scaled_features['excl'][1]),
            'return_pct': str(return_pcts['excl'])
        }
    }
    return return_val

if __name__ == '__main__':
    app.run(debug=True)
