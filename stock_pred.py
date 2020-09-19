import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#importing lib from keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
path1="data/AMZNtrain.csv"
#getting the data for dataprocessing
datasets=pd.read_csv(path1,index_col="Date",parse_dates=True)
#print(datasets.head(10))
tot=(len(datasets))

d1=datasets['Close']
plt.plot(datasets)
#minmaxscalar is used for normalisation
NM=MinMaxScaler(feature_range=(0,1))
datasets_scaled=NM.fit_transform(datasets)

######################################################################################

#creating datset for training
tot=len(datasets_scaled)
X_train=[]
y_train=[]
#creating with a structure of 60 timesteps
#we are going to analyse for close stock
for i in range(60, tot):
    X_train.append(datasets_scaled[i-60:i, 4])
    y_train.append(datasets_scaled[i, 4])

X_train, y_train = np.array(X_train), np.array(y_train)
#reshape 2d to 3d
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

######################################################################################

#initialise Rnn algorithm

regressor=Sequential()
#adding layers for LSTM and dropouts
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#adding layers for LSTM and dropouts
regressor.add(LSTM(units=50,return_sequences =True))
regressor.add(Dropout(0.2))
#adding layers for LSTM and dropouts
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#adding layers for LSTM and dropouts
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


#adding outer layer 
regressor.add(Dense(units=1))

#compile the rnn
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting the rnn to the training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


######################################################################################
#creating test datasets


path2='data/AMZNtest.csv'
dataset_test = pd.read_csv(path2,index_col="Date",parse_dates=True)
#print(len(dataset_test))
#print(dataset_test.head(5))
real_stock_price = dataset_test.iloc[:, 4].values
#print(len(real_stock_price ))
#print(real_stock_price)
# Getting the predicted stock price
dataset_total = pd.concat((datasets['Close'], dataset_test['Close']), axis = 0)
#print(len(dataset_total))
#print(dataset_total.head(5))
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)
#print(len(inputs))
inputs = NM.fit_transform(inputs)

#test dataset for predicting

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = NM.inverse_transform(predicted_stock_price)
predicted_stock_price=predicted_stock_price.reshape(-1,1)

########################################################################################

#plottiing the datasets  
plt.plot(real_stock_price, color = 'red', label = 'Real amazon Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted amazon Stock Price')
plt.title('Google amazon Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google amazon Price')

plt.show()
