# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:19:45 2020

@author: rayll
"""

"""Convolutional Neural Network"""

#para o pre-processamento:
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#para o treinamento:
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

#importando os dados
(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()

"""Pré-processamento"""
#ajustando o formato dos dados para a rede
train_X=train_X.reshape(-1,28,28,1)
test_X=test_X.reshape(-1,28,28,1)
#modificando para o formato float32
train_X=train_X.astype('float32')
test_X=test_X.astype('float32')
#reescalando os valores do pixel para variar entre 0 e 1
train_X=train_X/255
test_X=test_X/255
#transformando as categorias para one-hot encoding
train_Y_oneHot=to_categorical(train_Y)
test_Y_oneHot=to_categorical(test_Y)
#separando 20% dos dados de treinamento para a validaçao
train_X, valid_X, train_label, valid_label = train_test_split(train_X,train_Y_oneHot,train_size=0.8, test_size=0.2)

"""Gerando a CNN"""
batch_size=64
epochs=20
num_classes=10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.2)) #
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.2)) #
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.2)) #
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))       
fashion_model.add(Dropout(0.2)) #           
fashion_model.add(Dense(num_classes, activation='softmax'))
#compilando o modelo
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

"""Treinando o modelo"""
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

"""Testando o modelo"""
test_eval = fashion_model.evaluate(test_X, test_Y_oneHot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

"""Curvas de aprendizado"""
arqv=pd.DataFrame()
arqv['acuracia']=fashion_train.history['accuracy']
arqv['val_acuracia']=fashion_train.history['val_accuracy']
arqv['epocas']=range(len(arqv['acuracia']))
arqv['epocas']+=1
arqv.to_csv('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Exercicio04/CNN-fashion_mnist-acuracia.csv',index=False)
plt.style.use('seaborn')
ax=plt.axes()
plt.title('Curva de aprendizado',size=16)
plt.plot(arqv['epocas'],arqv['acuracia'],label='Treinamento')
plt.plot(arqv['epocas'],arqv['val_acuracia'],label='Validação',c='r')
plt.legend(fontsize=14)
ax.set(xlabel='Épocas',ylabel='Acurácia')
plt.savefig('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Exercicio04/CNN-fashion_mnist-curva_aprendizado.png',dpi=300)

"""Matriz de erro"""
test_predict=fashion_model.predict(test_X)
predict=[]
n=0
while n < len(test_predict):
    x=0
    while x < len(test_predict[n]):
        if test_predict[n][x]==max(test_predict[n]):
            predict.append(x)
        x+=1
    n+=1
mc=confusion_matrix(test_Y,predict)
mc=pd.DataFrame(mc)
mc.to_csv('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Exercicio04/CNN-fashion_mnist-matriz_confusao.csv')
print(mc)
