import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import r2_score
import pandas as pd

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def build_and_compile_model(input_dim=11):
    """
    Function to build and compile DNN architecture
    """
    model = keras.Sequential([
        layers.Dense(64, input_dim=input_dim, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def fit_model(model, epochs, batch_size, train_features, train_labels):
    """
    Function to fit the DNN model with specified epochs and batch_size
    """
    print(model.summary())
    history = model.fit(train_features,
                        train_labels, validation_split=0.2,
                        verbose=0, epochs=epochs, batch_size=batch_size)
    return history

def plot_loss(history):
    """
    Function to plot history's loss
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Close]')
    plt.legend()
    plt.grid(True)
    plt.show()

def genPredictions(model,ori_df,test_features,train_len):
    """
    Function to generate predictions with the developed model
    Returns predictions dataframe with Pred and Actual columns
    """
    predictions = model.predict(test_features).flatten()
    actual_close = ori_df[['Close']]
    
    act = pd.DataFrame(actual_close.iloc[train_len:, 0])
    
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = test_features.index
    predictions['Actual'] = act['Close']
    predictions.rename(columns={0:'Pred'}, inplace=True)
    return predictions

def plotPredAct(predictions_df):
    """
    Function to plot predictions versus actual values
    """
    predictions_df['Actual'].plot(figsize=(20,8), legend=True, color='blue')
    predictions_df['Pred'].plot(legend=True, color='red', figsize=(20,8))

def inversePredsAndAct(predictions_df, close_scaler, test_labels):
    """
    Function to inverse transform the predicted and actual values
    """
    inversed_pred = close_scaler.inverse_transform(np.array(predictions_df['Pred']).reshape(-1,1))
    inversed_act = close_scaler.inverse_transform(np.array(predictions_df['Actual']).reshape(-1,1))
    
    inversed = pd.DataFrame(inversed_pred)
    inversed['Actual'] = inversed_act
    inversed.rename({0:'Pred'}, axis=1, inplace=True)
    inversed.index = test_labels.index
    
    return inversed

def plotErrorHist(inversed_df):
    """
    Function to plot error histogram
    """
    error = inversed_df['Pred'] - inversed_df['Actual']
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Close]')
    _ = plt.ylabel('Count')

def evaluateModel(inversed_df):
    rmse_ = rmse(inversed_df['Pred'], inversed_df['Actual'])
    mape_ = mape(inversed_df['Actual'], inversed_df['Pred'])
    rsquared_ = r2_score(inversed_df['Actual'], inversed_df['Pred'])
    return rmse_, mape_, rsquared_