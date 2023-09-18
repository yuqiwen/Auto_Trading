import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model
from matplotlib.ticker import FuncFormatter
from math import sqrt
import yfinance as yf
import pandas as pd
from keras.models import load_model


# show plot result
def plot_results(predicted_data, true_data,company,dates):
    def dollar_formatter(x, pos):
        return f"${x:,.0f}"
    fig = plt.figure(figsize=(20,15),facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(dates,true_data, label='True Data')
    plt.plot(dates,predicted_data, label='Predicted Data')
    ax.set_title(f"Stock Prices and Prediction for {company}", fontsize=20, color='blue')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    # plt.xlabel('{}'.format(name))
    plt.legend()
    #plt.show()
    plt.savefig('results.png')

#RNN time series
def main():
    # Set the working directory
    #os.chdir("your/directory/to/ensure/file/read")
    #Read input
    # company="GOOG"
    id=input("Please enter company: ")
    company=yf.Ticker(id)
    df=company.history(period='max',interval="1d")
    df.to_csv('data/stock.csv')
    
    # # Load the model
    # model = load_model('path_to_saved_model.h5')

    # # Continue training if needed
    # model.fit(train_data, train_labels, ...)

    # # Or use it for predictions
    # predictions = model.predict(new_data)

    configs = json.load(open('config_1.json', 'r'))
    # if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    #Load data
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    #create RNN model
    model = Model()
    mymodel = model.build_model(configs)
    # plot_model(mymodel, to_file='model.png',show_shapes=True)
    
    #get train data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    # print (x.shape)
    # print (y.shape)
    
	#train model
    model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	
   #get test data
    x_test, y_test,data_windows= data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    #show prediction result
    predictions_pointbypoint = model.predict_point_by_point(x_test,debug=True)        

    
    denormalised_predictions = data.denormalise_windows(data.data_test[configs['data']['sequence_length']:], predictions_pointbypoint)
    denormalised_y_test = data.denormalise_windows(data.data_test[configs['data']['sequence_length']:], y_test)
    length=len(denormalised_y_test)
    date_list = [timestamp.date() for timestamp in df.index]

    dates=date_list[-length:]

    plot_results(denormalised_predictions,denormalised_y_test,id,dates)

    #Predict Price of tomorrow
    last_window = data_windows[-1][np.newaxis, ...]  # Shape will be (1, 49, features)
    last_real_price = data.data_test[-1]
    t = model.predict_point_by_point(last_window,debug=True) 
    tomorrow=last_real_price*np.exp(t)
    print("Tomorrow's Price Predicted:",float(tomorrow[0]))

    
if __name__ == '__main__':
    main()
