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
    model.load_model(f'../experiments_robust/XGBReg/models/{company}.json')
    
    # Generate Features
    ticker = company.upper()+'.JK'
    fg = FeatureGenerator(ticker=ticker)
    scaled_features = fg.scaled_features

    # Predict
    prediction = model.predict(scaled_features)

    # Inverse predicted
    close_scaler = load(f'../experiments_robust/feature_engineering/{company}_close_scaler.bin')
    inversed = close_scaler.inverse_transform(np.array(prediction).reshape(-1,1))

    return inversed.tolist()[0]

if __name__ == '__main__':
    app.run(debug=True)
