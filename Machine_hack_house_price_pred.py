import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('D:\ML_projects\Machine Hack\Participants_Data_HPP\Train.csv')
dataset_test = pd.read_csv('D:\ML_projects\Machine Hack\Participants_Data_HPP\Test.csv')

def preprocess(dataset):
    data = dataset.iloc[:].values
    
    if dataset.shape[1] == 12:
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)
    else:
        X = data[:, :-1]
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    label_encoder = LabelEncoder()
    X[:, 0] = label_encoder.fit_transform(X[:, 0])
    X[:, 1] = label_encoder.fit_transform(X[:, 1])
    # X[:, 4] = label_encoder.fit_transform(X[:, 4])
    
    onehotencoder = OneHotEncoder()
    
    # onehotencoded_columns = onehotencoder.fit_transform(X[:, 4].reshape(-1, 1)).toarray()
    # X = np.delete(X, 4, 1)
    # X = np.insert(X, [4], onehotencoded_columns[:], axis = 1)
    onehotencoded_columns = onehotencoder.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
    X = X[:, 1:]
    X = np.insert(X, [0], onehotencoded_columns[:, 1:], axis = 1)
    if dataset.shape[1] == 12:
        return(X, Y)
    else:
        return(X)

X_train, Y_train = preprocess(dataset_train)
X_test = preprocess(dataset_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train.reshape(-1, 1))

# from sklearn.preprocessing import MinMaxScaler
# mm_x = MinMaxScaler()
# X_train = mm_x.fit_transform(X_train)
# X_test = mm_x.transform(X_test)
# mm_y = MinMaxScaler()
# Y_train = mm_y.fit_transform(Y_train.reshape(-1, 1))
    

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers as k_o
from keras import initializers as k_i
# from keras.layers import LSTM

# INIT_LR = 0.0005

# regressor = Sequential()
# # regressor.add(LSTM(64, activation = 'relu', input_shape = (X_train.shape[1], 9),
# #                    return_sequences = True))
# # regressor.add(LSTM(64, activation = 'relu'))
# regressor.add(Dense(128, activation = 'relu', input_shape = (12, )))
# # regressor.add(Dropout(0.2))
# regressor.add(Dense(128, activation = 'relu'))
# # regressor.add(Dropout(0.2))
# regressor.add(Dense(128, activation = 'relu'))
# # regressor.add(Dropout(0.2))
# # regressor.add(Dense(128, activation = 'relu'))
# # regressor.add(Dropout(0.1))
# # regressor.add(Dense(128, activation = 'relu'))
# # regressor.add(Dropout(0.1))
# # regressor.add(Dense(128, activation = 'relu'))
# # regressor.add(Dropout(0.1))
# # regressor.add(Dense(128, activation = 'relu'))
# regressor.add(Dense(1))

# opt = k_o.Adam(lr = INIT_LR, decay = 0.000002)

# regressor.compile(optimizer = opt, loss = 'mean_squared_error', 
#                   metrics = ['mean_squared_error'])

# history = regressor.fit(X_train, Y_train, batch_size = 128, 
#                         epochs = 300, validation_split = 0.5)

# Y_pred_train_1 = regressor.predict(X_train)
# Y_pred_test_1 = regressor.predict(X_test)

# X_train = np.insert(X_train, [X_train.shape[1]], Y_pred_train_1, axis = 1)

# X_test = np.insert(X_test, [X_test.shape[1]], Y_pred_test_1, axis = 1)


INIT_LR = 0.0005

# X_train = X_train.reshape(-1, len(X_train), 9)
# Y_train = Y_train.reshape(-1, len(Y_train), 9)

regressor = Sequential()
# regressor.add(LSTM(64, activation = 'relu', input_shape = (X_train.shape[1], 9),
#                    return_sequences = True))
# regressor.add(LSTM(64, activation = 'relu'))
regressor.add(Dense(1024, activation = 'relu', input_shape = (11, )))
                    # kernel_initializer = k_i.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
# regressor.add(Dropout(0.2))
regressor.add(Dense(64, activation = 'relu'))
# regressor.add(Dropout(0.2))
# regressor.add(Dense(64, activation = 'relu'))
# regressor.add(Dropout(0.2))
# regressor.add(Dense(64, activation = 'relu'))
# regressor.add(Dropout(0.1))
# regressor.add(Dense(64, activation = 'relu'))
# regressor.add(Dropout(0.1))
# regressor.add(Dense(128, activation = 'relu'))
# regressor.add(Dropout(0.1))
# regressor.add(Dense(128, activation = 'relu'))
regressor.add(Dense(1))

opt = k_o.Adam(lr = INIT_LR, decay = 0.000001)

regressor.compile(optimizer = opt, loss = 'mean_squared_error', 
                  metrics = ['mean_squared_error'])

history = regressor.fit(X_train, Y_train, batch_size = 128, 
                        epochs = 200, validation_split = 0.2)

Y_pred = regressor.predict(X_test)
Y_pred = sc_y.inverse_transform(Y_pred)

plt.close("all")
# plt.figure(1)
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

for i in Y_pred:
    if i<0:
        print('negative values')

row = ['TARGET(PRICE_IN_LACS)']

import csv
# filename = "D:\ML_projects\Machine Hack\Participants_Data_HPP\results.csv"
  
# writing to csv file 
with open('results.csv', 'w', newline='') as file:
    csvwriter = csv.writer(file) 
      
    # writing the fields 
    csvwriter.writerow(row) 
      
    # writing the data rows 
    csvwriter.writerows(Y_pred)
