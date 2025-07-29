#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing packages #
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# neglect warning #
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('Group11_FinalReport_DS2_2022.csv')
df = df.iloc[2: , :]


# In[ ]:


df=df[df.columns]
df.shape



# Renaming of the columns with powerplant terms

# In[ ]:


df.rename(columns={'1CHC00GH902':'GENERATOR MW',
'NumDisp.9882':'TOTAL COAL FLOW BEF CAL CORR',
'MPS02-MSS-TAF-03':'TOTAL AIR FLOW(t/h)',
'1HAD01CP902':'FURNACE DRAFT PRESS',
'1HLA51CG771XQ01':'AA (L) CRNR 1 AA DMPR  FB',
'1HLA52CG771XQ01':'AA (L) CRNR 2 AA DMPR  FB',
'1HLA53CG771XQ01':'AA (L) CRNR 3 AA DMPR FB',
'1HLA54CG771XQ01':'AA (L) CRNR 4 AA DMPR FB',
'1HLA55CG771XQ01':'AA (L) CRNR 5 AA DMPR FB',
'1HLA56CG771XQ01':'AA (L) CRNR 6 AA DMPR FB',
'1HLA57CG771XQ01':'AA (L) CRNR 7 AA DMPR FB',
'1HLA58CG771XQ01':'AA (L) CRNR 8 AA DMPR FB',
'MPS01-WFR2-02-01':'WTR FUEL RATIO',
'1LAE01CF901_CP':'1RY SH DSH FW FLOW',
'1LAE01CF902':'2RY SH DSH FW FLOW',
'Unnamed: 15':'RH ECO O/L FLUE GAS DMPR',
'Unnamed: 16':'SH ECO O/L FLUE GAS DMPR ',
'1HAD01CP157XQ01':'WIND BOX and FURNACE DP 1',
'1HAD01CP158XQ01':'WIND BOX and FURNACE DP 2',
'1HHA51CG701XQ01':'1CORNER BURNER TILTING DRIVE  FB(deg)',
'1HHA52CG701XQ01':'2CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA53CG701XQ01':'3CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA54CG701XQ01':'4CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA55CG701XQ01':'5CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA56CG701XQ01':'6CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA57CG701XQ01':'7CORNER BURNER TILTING DRIVE FB(deg)',
'1HHA58CG701XQ01':'8CORNER BURNER TILTING DRIVE FB(deg)',
'1LBA01CT911':'MS LINE A TEMP',
'1LBA01CT951':'MS LINE B TEMP',
'1HAD01CT906':'WW O/L RHT TEMP',
'1HAD01CT956':'WW O/L LFT TEMP',
'1HLA51CG712XQ01':'COAL BNR A CRNR 1 CNTRL DMPR POS',
'1HLA51CG712XQ02':'COAL BNR A CRNR 1 CNTRL DMPR POS.1',
'1HLA51CG712XQ03':'COAL BNR A CRNR 1 CNTRL DMPR POS.2',
'1HLA51CG712XQ04':'COAL BNR A CRNR 1 CNTRL DMPR POS.3',
'1HLA51CG712XQ05':'COAL BNR A CRNR 1 CNTRL DMPR POS.4',
'1HLA51CG712XQ06':'COAL BNR A CRNR 1 CNTRL DMPR POS.5',
'1HLA51CG712XQ07':'COAL BNR A CRNR 1 CNTRL DMPR POS.6',
'1HLA51CG712XQ08':'COAL BNR A CRNR 1 CNTRL DMPR POS.7',
'1HLA51CG712XQ09':'COAL BNR A CRNR 1 CNTRL DMPR POS.8',
'1HLA51CG712XQ10':'COAL BNR A CRNR 1 CNTRL DMPR POS.9',
'1HLA51CG712XQ11':'COAL BNR A CRNR 1 CNTRL DMPR POS.10',
'1HLA51CG712XQ12':'COAL BNR A CRNR 1 CNTRL DMPR POS.11',
'1HLA51CG712XQ13':'COAL BNR A CRNR 1 CNTRL DMPR POS.12',
'1HLA51CG712XQ14':'COAL BNR A CRNR 1 CNTRL DMPR POS.13',
'1HLA51CG712XQ15':'COAL BNR A CRNR 1 CNTRL DMPR POS.14',
'1HLA51CG712XQ16':'COAL BNR A CRNR 1 CNTRL DMPR POS.15',
'1HLA51CG712XQ17':'COAL BNR A CRNR 1 CNTRL DMPR POS.16',
'1HLA51CG712XQ18':'COAL BNR A CRNR 1 CNTRL DMPR POS.17',
'1HLA51CG712XQ19':'COAL BNR A CRNR 1 CNTRL DMPR POS.18',
'1HLA51CG712XQ20':'COAL BNR A CRNR 1 CNTRL DMPR POS.19',
'1HLA51CG712XQ21':'COAL BNR A CRNR 1 CNTRL DMPR POS.20',
'1HLA51CG712XQ22':'COAL BNR A CRNR 1 CNTRL DMPR POS.21',
'1HLA51CG712XQ23':'COAL BNR A CRNR 1 CNTRL DMPR POS.22',
'1HLA51CG712XQ24':'COAL BNR A CRNR 1 CNTRL DMPR POS.23',
'1HLA51CG712XQ25':'COAL BNR A CRNR 1 CNTRL DMPR POS.24',
'1HLA51CG712XQ26':'COAL BNR A CRNR 1 CNTRL DMPR POS.25',
'1HLA51CG712XQ27':'COAL BNR A CRNR 1 CNTRL DMPR POS.26',
'1HLA51CG712XQ28':'COAL BNR A CRNR 1 CNTRL DMPR POS.27',
'1HLA51CG712XQ29':'COAL BNR A CRNR 1 CNTRL DMPR POS.28',
'1HLA51CG712XQ30':'COAL BNR A CRNR 1 CNTRL DMPR POS.29',
'1HLA51CG712XQ31':'COAL BNR A CRNR 1 CNTRL DMPR POS.30',
'1HLA51CG712XQ32':'COAL BNR A CRNR 1 CNTRL DMPR POS.31',
'1HLA51CG712XQ33':'COAL BNR A CRNR 1 CNTRL DMPR POS.32',
'1HLA51CG712XQ34':'COAL BNR A CRNR 1 CNTRL DMPR POS.33',
'1HLA51CG712XQ35':'COAL BNR A CRNR 1 CNTRL DMPR POS.34',
'1HLA51CG712XQ36':'COAL BNR A CRNR 1 CNTRL DMPR POS.35',
'1HLA51CG712XQ37':'COAL BNR A CRNR 1 CNTRL DMPR POS.36',
'1HLA51CG712XQ38':'COAL BNR A CRNR 1 CNTRL DMPR POS.37',
'1HLA51CG712XQ39':'COAL BNR A CRNR 1 CNTRL DMPR POS.38',
'1HLA51CG712XQ40':'COAL BNR A CRNR 1 CNTRL DMPR POS.39',
'1HLA51CG712XQ41':'COAL BNR A CRNR 1 CNTRL DMPR POS.40',
'1HLA51CG712XQ42':'COAL BNR A CRNR 1 CNTRL DMPR POS.41',
'1HLA51CG712XQ43':'COAL BNR A CRNR 1 CNTRL DMPR POS.42',
'1HLA51CG712XQ44':'COAL BNR A CRNR 1 CNTRL DMPR POS.43',
'1HLA51CG712XQ45':'COAL BNR A CRNR 1 CNTRL DMPR POS.44',
'1HLA51CG712XQ46':'COAL BNR A CRNR 1 CNTRL DMPR POS.45',
'1HLA51CG712XQ47':'COAL BNR A CRNR 1 CNTRL DMPR POS.46',
'1HLA51CG712XQ48':'COAL BNR A CRNR 1 CNTRL DMPR POS.47',
'1HLA51CG713XQ01':'HFO/LDO BNR AB CRNR 1 DMPR POS',
'1HLA52CG713XQ01':'HFO/LDO BNR AB CRNR 2 CNTRL DMPR POS',
'1HLA53CG713XQ01':'HFO/LDO BNR AB CRNR 3 DMPR POS',
'1HLA54CG713XQ01':'HFO/LDO BNR AB CRNR 4 CNTRL DMPR POS',
'1HLA55CG713XQ01':'HFO/LDO BNR AB CRNR 5 CNTRL DMPR POS',
'1HLA56CG713XQ01':'HFO/LDO BNR AB CRNR 6 CNTRL DMPR POS',
'1HLA57CG713XQ01':'HFO/LDO BNR AB CRNR 7 CNTRL DMPR POS',
'1HLA58CG713XQ01':'HFO/LDO BNR AB CRNR 8 CNTRL DMPR POS',
'1HLA51CG733XQ01':'HFO BNR CD CRNR 1 CNTRL DMPR POS',
'1HLA52CG733XQ01':'HFO BNR CD CRNR 2 CNTRL DMPR POS',
'1HLA53CG733XQ01':'HFO BNR CD CRNR 3 DMPR POS',
'1HLA54CG733XQ01':'HFO BNR CD CRNR 4 CNTRL DMPR POS',
'1HLA55CG733XQ01':'HFO BNR CD CRNR 5 CNTRL DMPR POS',
'1HLA56CG733XQ01':'HFO BNR CD CRNR 6 CNTRL DMPR POS',
'1HLA57CG733XQ01':'HFO BNR CD CRNR 7 CNTRL DMPR POS',
'1HLA58CG733XQ01':'HFO BNR CD CRNR 8 CNTRL DMPR POS',
'1HLA51CG753XQ01':'HFO BNR EF CRNR 1CNTRL DMPR POS',
'1HLA52CG753XQ01':'HFO BNR EF CRNR 2 CNTRL DMPR POS',
'1HLA54CG753XQ01':'HFO BNR EF CRNR 4 CNTRL DMPR POS',
'1HLA55CG753XQ01':'HFO BNR EF CRNR 5 CNTRL DMPR POS',
'1HLA56CG753XQ01':'HFO BNR EF CRNR 6 CNTRL DMPR POS',
'1HLA56CG753XQ01.1':'HFO BNR EF CRNR 6 CNTRL DMPR POS.1',
'1HLA57CG753XQ01':'HFO BNR EF CRNR 7 CNTRL DMPR POS',
'1HLA58CG753XQ01':'HFO BNR EF CRNR 8 CNTRL DMPR POS',
'1HAH02CT901':'2RY SH A I/L STM TEMP',
'1HAH02CT903':'2RY SH B I/L STM TEMP',
'1HAH02CT902':'2RY SH A O/L TEMP',
'1HAH02CT904':'2RY SH B O/L TEMP',
'1HAH03CT901':'3RY SH A I/L TEMP',
'1HAH03CT902':'3RY SH B I/L TEMP',
'1HAJ01CT001XQ01':'1RY RHTR O/L TEMP 1',
'1HAJ01CT051XQ01':'1RY RHTR O/L TEMP 2',
'1HAJ02CT901':'2RY RH-A I/L STM TEMP ',
'1HAJ02CT951':'2RY RH-B I/L STM TEMP',
'1LBB01CT902':'2RY RH-A O/L TEMP',
'1LBB01CT951':'2RY RH-B O/L TEMP',
'1LAF01CF151XQ01':'RH DSH FW FLOW'},inplace = True)


# In[ ]:


df.head()


# Removing the outliers:

# In[ ]:


df = df.astype(float)
df['GENERATOR MW'].plot()
plt.ylim((600, 725))


# In[ ]:


for col in df.columns:
  mean = df[col].mean()
  median = df[col].median()
  std = df[col].std()
  df[col]= np.where(((df[col] - mean)/std).abs()>3,np.nan,df[col])
  df[col].fillna(method='ffill', inplace=True)



# After removing outliers:

# In[ ]:


df['GENERATOR MW'].plot()
plt.ylim((600, 725))


# In[ ]:


for col in df.columns:
  print('for column {0} {1}'.format(col, df[col].isnull().values.any()))


# To check if the dataset contains Nan/Null values

# In[ ]:


df.isnull().sum().sum()


# In[ ]:


df = df.astype( str)
df.dtypes


# To check if the dataset contains any special characters

# In[ ]:


for col in df.columns:
          print(df[df[col].str.contains(r'[@&$%+_]')])




# In[ ]:


df = df.astype(float)
df.dtypes
df.to_csv('Data_cleaned.csv', index=False)


# Inputs and Outputs selection:

# In[ ]:


df.head()
x1 = df.copy()
x1.head()


# In[ ]:


y = df.iloc[:,103:116]
y['GENERATOR MW'] = df.iloc[:,0]
y['MS LINE A TEMP'] = df['MS LINE A TEMP']
y['MS LINE B TEMP'] =df['MS LINE B TEMP']
y['WW O/L RHT TEMP'] =df['WW O/L RHT TEMP']
y['WW O/L LFT TEMP']= df['WW O/L LFT TEMP']
y.shape


# In[ ]:


for col in y.columns:
  x1.drop(columns=[col],inplace=True)


# In[ ]:


x1.head()


# In[ ]:


X =x1
X = X.astype('float32')
y = y.astype(float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y)
X = X.astype('float32')


# In[ ]:


from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
series = df
generator_values = series['GENERATOR MW']
val2 = series['MS LINE A TEMP']
val3 = series['2RY RH-A O/L TEMP']
# generator_values = generator_values.drop(labels=[0,1], axis=0)
plot_acf(generator_values, lags=50)
pyplot.title('Autocorrelation plot for Generator MW')
pyplot.savefig('ACF plots for Generator MW.png')
plot_acf(val2, lags=50)
pyplot.title('Autocorrelation plot for Main steam temperature')
pyplot.savefig('ACF plots for Main steam temperature.png')
pyplot.show()


# Data spliting:
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


# Deep Neural Network:

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
import tensorflow as tf



# In[ ]:


# different structures of network have been checked , this one yields the optimal result #
# Neural network
model = Sequential()
model.add(Dense(60, input_dim=98, activation='sigmoid'))#input layer 
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(18,  activation=tf.keras.activations.linear)) #output layer
model.summary()


# In[ ]:


import tensorflow as tf
from keras import backend as K

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt,  metrics=[R2])


# In[ ]:


history1 = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=200, batch_size= 500)


# In[ ]:


plt.plot(history1.history['R2'])
plt.plot(history1.history['val_R2'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='bottom right')
plt.show()


# In[ ]:


plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='bottom right')
plt.show()


# In[ ]:


y_pred = model.predict(X_test)
plt.plot(y_pred[:,13])
plt.plot(y_test[:,13])
plt.xlim([0,100])


# In[ ]:


y_a = sc.inverse_transform(y_test)[:, [13]]
y_t = sc.inverse_transform(y_pred)[:, [13]]
plt.plot(y_t)
plt.plot(y_a)
plt.xlim([0,100])


# since the loss was still higher we proceeded with CNN & RNN further.

# Energy optimization using Convolutional Neural Network (CNN)
# 

# In[ ]:


import tensorflow as tf
from keras import backend as K

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# Simple CNN architecture

# In[ ]:


model_2 = Sequential()
model_2.add(tf.keras.layers.Conv1D(32, kernel_size=8, activation='relu', padding='same', input_shape=(98, 1)))
model_2.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model_2.add(tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'))
model_2.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(64, activation='relu'))
model_2.add(tf.keras.layers.Dense(18))
model_2.summary()
model_2.compile(loss="mse", optimizer="Adam", metrics=[R2])


# In[ ]:


history_2 = model_2.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=200)


# In[ ]:


test_loss, test_acc = model_2.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history_2.history['loss'], label='training loss')
plt.plot(history_2.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0, 10)
plt.legend()


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history_2.history['R2'], label='training accuracy')
plt.plot(history_2.history['val_R2'], label='testing accuracy')
plt.title('Accuracy vs number of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
# plt.ylim(-15, 1)
plt.legend()


# In[ ]:


# X_t = sc.inverse_transform(X_test)
y_t = sc.inverse_transform(y_test)
X_limited_pts = X_test[2700:]
y_limited_pts = y_t[2700:]
# y_limited_pts_arr = y_limited_pts.to_numpy()
X_limited_pts.shape


# In[ ]:


y_pred = model_2.predict(X_limited_pts)
y_pred = sc.inverse_transform(y_pred)
plt.plot(y_pred[:,13])
plt.plot(y_limited_pts[:,13])
plt.title('Predicted vs Actual values')
plt.xlabel('Observations')
plt.ylabel('Generator MW')
plt.legend()
plt.savefig('Predicted vs actual values for Generator MW')


# In[ ]:


y_pred = model_2.predict(X_limited_pts)
y_pred = sc.inverse_transform(y_pred)
plt.plot(y_pred[:,15])
plt.plot(y_limited_pts[:,15])
plt.title('Predicted vs Actual values')
plt.xlabel('Observations')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('Predicted vs actual values for Temperature')


# ResNet architecture for Conv1D

# In[ ]:


from keras.utils.vis_utils import plot_model
input_layer = tf.keras.layers.Input(shape=(98, 1))

# BLOCK 1

conv_x = tf.keras.layers.Conv1D(filters=64, kernel_size=8, padding='same')(input_layer)
conv_x = tf.keras.layers.BatchNormalization()(conv_x)
conv_x = tf.keras.layers.Activation('relu')(conv_x)
conv_y = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(conv_x)
conv_y = tf.keras.layers.BatchNormalization()(conv_y)
conv_y = tf.keras.layers.Activation('relu')(conv_y)
conv_z = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv_y)
conv_z = tf.keras.layers.BatchNormalization()(conv_z)
shortcut_residuals = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='same')(input_layer)
shortcut_residuals = tf.keras.layers.BatchNormalization()(shortcut_residuals)
output_block_1 = tf.keras.layers.add([shortcut_residuals, conv_z])
output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

# BLOCK 2

conv_x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(output_block_1)
conv_x = tf.keras.layers.BatchNormalization()(conv_x)
conv_x = tf.keras.layers.Activation('relu')(conv_x)
conv_y = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv_x)
conv_y = tf.keras.layers.BatchNormalization()(conv_y)
conv_y = tf.keras.layers.Activation('relu')(conv_y)
conv_z = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv_y)
conv_z = tf.keras.layers.BatchNormalization()(conv_z)
shortcut_residuals = tf.keras.layers.Conv1D(filters=128, kernel_size=1, padding='same')(output_block_1)
shortcut_residuals = tf.keras.layers.BatchNormalization()(shortcut_residuals)
output_block_2 = tf.keras.layers.add([shortcut_residuals, conv_z])
output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

# BLOCK 3

conv_x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(output_block_2)
conv_x = tf.keras.layers.BatchNormalization()(conv_x)
conv_x = tf.keras.layers.Activation('relu')(conv_x)
conv_y = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv_x)
conv_y = tf.keras.layers.BatchNormalization()(conv_y)
conv_y = tf.keras.layers.Activation('relu')(conv_y)
conv_z = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv_y)
conv_z = tf.keras.layers.BatchNormalization()(conv_z)
shortcut_residuals = tf.keras.layers.BatchNormalization()(output_block_2)
output_block_3 = tf.keras.layers.add([shortcut_residuals, conv_z])
output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)
gap_layer = tf.keras.layers.GlobalAveragePooling1D()(output_block_3)
output_layer = tf.keras.layers.Dense(18)(gap_layer)
model_9 = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
opt = tf.keras.optimizers.Adam()
model_9.compile(optimizer=opt, loss='mse', metrics=[R2])
plot_model(model_9, to_file='resnet_model_plot_another.png', show_shapes=True, show_layer_names=True)


# In[ ]:


history_9 = model_9.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=200)


# In[ ]:


test_loss, test_acc = model_9.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history_9.history['loss'], label='training loss')
plt.plot(history_9.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0, 10)
plt.legend()
plt.savefig('Training and testing loss for ResNet CNN.pdf')


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history_9.history['R2'], label='training accuracy')
plt.plot(history_9.history['val_R2'], label='testing accuracy')
plt.title('Accuracy vs number of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy')
# plt.ylim(-15, 1)
plt.legend()
plt.savefig('Training and testing R2 scores for ResNet CNN.pdf')


# In[ ]:


# X_t = sc.inverse_transform(X_test)
y_t = sc.inverse_transform(y_test)
X_limited_pts = X_test[5100:]
y_limited_pts = y_t[5100:]
# y_limited_pts_arr = y_limited_pts.to_numpy()
X_limited_pts.shape


# In[ ]:


y_pred = model_9.predict(X_limited_pts)
y_pred = sc.inverse_transform(y_pred)
plt.plot(y_pred[:,13])
plt.plot(y_limited_pts[:,13])
plt.title('Predicted vs Actual values')
plt.xlabel('Observations')
plt.ylabel('Generator MW')
plt.legend()
plt.savefig('Predicted vs actual values for Generator MW')


# In[ ]:


y_pred = model_9.predict(X_limited_pts)
y_pred = sc.inverse_transform(y_pred)
plt.plot(y_pred[:,15])
plt.plot(y_limited_pts[:,15])
plt.title('Predicted vs Actual values')
plt.xlabel('Observations')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('Predicted vs actual values for Temperature')


# Recurrent Neural Network:
# 

# In[ ]:


X_RNN = X.reshape(17280, 1, 98)
print(X_RNN.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_RNN,y,test_size = 0.3,random_state=42)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
model = Sequential()
model.add(LSTM(256, input_shape=X_RNN.shape[1:], activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, input_shape=X_RNN.shape[1:], activation='relu'))
model.add(Dropout(0.1))

# model.add(LSTM(256, input_shape=X_RNN.shape[1:], activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(18, activation=tf.keras.activations.linear)) #output layer


# In[ ]:


from keras import backend as K

def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# In[ ]:


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='mse',
    optimizer=opt,
    metrics=R2,
)

history=model.fit(X_train,
          y_train, batch_size=500,
          epochs=200,
          validation_data=(X_test, y_test))


# In[ ]:


from sklearn.metrics import r2_score, mean_absolute_error
train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
mo


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['R2'], label='training_R2 score')
plt.plot(history.history['val_R2'], label='testing_R2 score')
#plt.title('R2 score')
plt.xlabel('Epochs')
plt.ylabel('R2 score')
plt.savefig('output.pdf', dpi=300, bbox_inches='tight')

plt.ylim(0,1.1)
plt.legend()


# In[ ]:


plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
#plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputloss.pdf', dpi=300, bbox_inches='tight')


# In[ ]:


a=sc.inverse_transform(y_pred)
b=sc.inverse_transform(y_test)

plt.plot(a[:, 13])
plt.plot(b[:, 13])
plt.xlim(0,100)

plt.xlabel('Observations')
plt.ylabel('Generator output (MW)')

plt.savefig('MW.pdf', dpi=300, bbox_inches='tight')


# In[ ]:


a=sc.inverse_transform(y_pred)
b=sc.inverse_transform(y_test)

plt.plot(a[:, 15])
plt.plot(b[:, 15])
plt.xlim(0,100)

plt.xlabel('Observations')
plt.ylabel('MS LINE B TEMP(Celsius)')

plt.savefig('TEMP(Celsius).pdf', dpi=300, bbox_inches='tight')


# End
