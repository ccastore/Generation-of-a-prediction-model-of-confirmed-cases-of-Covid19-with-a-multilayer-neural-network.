#-------------------Import modules-----------------------

from google.colab import drive
drive.mount('/content/drive')
import numpy as np
from collections import Counter
import tensorflow as tf #version 1.x
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers

#--------------------Load the file------------------------
fechas=np.arange(3,81)
data=np.genfromtxt("/content/drive/My Drive/Covid19/time_series_covid19_confirmed_global.csv", delimiter=',', skip_header=1,usecols=(fechas))
print(data)

#--------------Show the plot of all data set--------------
import matplotlib.pyplot as plt
plt.figure(figsize = (20,12))
for i in range (len(data)):
  plt.plot(data[i,0:],"-")

#--------------sort the dataset---------------------------
numero_previos=15
datos=np.zeros((10473,numero_previos+2))

h=0
for p in range (data.shape[0]):
  pais=data[p,0:]
  dias_activos=0

  for i in range (len(pais)):
    previos=np.zeros(numero_previos)

    dato_fecha=pais[i]
    if dato_fecha>0:
      dias_activos=dias_activos+1

    for m in range (-(numero_previos),0):
      if i+m>=0:
        previos[m]=pais[i+m]
      else:
        previos[m]=0
    if dato_fecha>0:
      for w in range (numero_previos):
        datos[h][w]=previos[w]
      datos[h][numero_previos]=dias_activos
      datos[h][numero_previos+1]=dato_fecha
      print(datos[h])
      h=h+1

print(np.max(datos[:,15]))
datos_ordenados=(datos[datos[:,15].argsort()])
print(datos_ordenados)
print(datos_ordenados.shape)


#-----------------Standardize data-------------------------
#encontrar los valores maximos por columna de X_training
norm=np.zeros(datos_ordenados.shape[1])

for columna in range (datos_ordenados.shape[1]):
  norm[columna]=np.max(np.abs(datos_ordenados[:,columna]))
  print(norm[columna])

print (norm)

#dividir cada columna entre el maximo de la columna 
datos_norm=np.zeros((datos_ordenados.shape[0],datos_ordenados.shape[1]))

for columna in range(datos_ordenados.shape[1]):
  for fila in range (datos_ordenados.shape[0]):
    datos_norm[fila][columna]=datos_ordenados[fila][columna]/norm[columna]
    
print(datos_norm)

x=datos_norm[:,0:datos_norm.shape[1]-1]
y=datos_norm[:,datos_norm.shape[1]-1]

print(x.shape,y.shape)
print (y)

#------------------------Training----------------------------
model = Sequential()
model.add(Dense(60,input_shape=(x.shape[1],),activation='sigmoid'))
model.add(Dense(40,activation='sigmoid'))
model.add(Dense(1,activation='softplus'))


adam=optimizers.Adam(lr=.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mse'])

snn=model.fit(x,y,batch_size=100,nb_epoch=200 ,shuffle=True)

snn_pred = model.predict(x, batch_size=100)
print(snn_pred.shape)

plt.figure(figsize = (20,12))
plt.title("Casos de Covid-19 en ...",size=28)
plt.xlabel("fecha",size=28)
plt.ylabel("Numero de casos",size=28)
plt.grid()
plt.plot(snn_pred*norm[16],"*b") #16 is México
plt.plot(y*norm[16],".r")

#-------------------------Save model---------------------------
model.save("/content/drive/My Drive/Covid19/modelo.h5")
model.save_weights("/content/drive/My Drive/Covid19/pesos.h5")
