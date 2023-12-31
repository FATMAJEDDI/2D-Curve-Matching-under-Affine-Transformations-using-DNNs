#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense, Lambda, concatenate, Conv2D, MaxPooling2D, Dropout, Flatten

import matplotlib.pyplot as plt


# In[2]:


#R-squared est une métrique de performance utilisée pour évaluer les modèles de régression, pas une fonction de perte.


# In[3]:


dataframe = pd.read_csv('MPEG7_DBLabled.csv',header = None)
dataframe.head()
#dataframe.tail()


# In[4]:


print(dataframe)


# In[5]:


dataframe = pd.DataFrame(dataframe).to_numpy()
print(dataframe.shape)
print(dataframe)


# In[6]:


import random
random.seed(100)
random.shuffle(dataframe)
print(dataframe.shape)


# In[7]:


Current_contours=dataframe[:,0:200]
#print(contours[1,0:200])
print(Current_contours.shape)
post_Contours=dataframe[:,200:400]
print(post_Contours.shape)


# In[8]:


Transformation_matrix=dataframe[:,400:406]
print(Transformation_matrix[0,:])
print(Transformation_matrix.shape)


# In[9]:


#print classes and subclasses\n",
Subclass_class=dataframe[:,406:408]
print(Subclass_class.astype("int"))


# In[10]:


Reshape_CurreCon = Current_contours.reshape((Current_contours.shape[0], 2,100,1))
print(Reshape_CurreCon.shape)
Reshape_PostCon = post_Contours.reshape((post_Contours.shape[0], 2,100,1))
print(Reshape_PostCon.shape)


# In[11]:


train_size = int(Reshape_CurreCon.shape[0] * 0.75)
print(train_size)
test_size = int(Reshape_CurreCon.shape[0] * 0.25)
print(test_size)


# In[12]:


data_left_train = Reshape_CurreCon[:train_size]
data_right_train = Reshape_PostCon[:train_size]
affine_matrices_train = Transformation_matrix[:train_size]


# In[13]:


data_left_test = Reshape_CurreCon[-test_size:]
data_right_test = Reshape_PostCon[-test_size:]
affine_matrices_test = Transformation_matrix[-test_size:]


# In[14]:


# Définition de la fonction pour calculer la distance L2 entre deux vecteurs
def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = tf.reduce_sum(tf.square(vector1 - vector2), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


# In[15]:


import tensorflow.keras.backend as K
shape= (2,100,1) 
#Les entrées des deux modele siamese
print("[INFO] building siamese network...")
input_1 =tf.keras.layers.Input(shape) 
input_2 =tf.keras.layers.Input(shape)
## Définir le modèle siamese profond
model =keras.Sequential()
# Ajout de couches de convolution et de pooling
model.add(keras.layers.Conv2D(16, kernel_size=(1,1), activation='relu',  padding='same', input_shape=shape, name="Conv2D_1"))
model.add(keras.layers.Conv2D(32, kernel_size=(1,1), activation='relu',  padding='same', name="Conv2D_2"))
model.add(keras.layers.Conv2D(64, kernel_size=(1,1), activation='relu',  padding='same', name="Conv2D_3"))
model.add(keras.layers.MaxPooling2D(pool_size=(1,1), name="MaxPooling2D_1"))


# Conversion en vecteur et ajout de couches fully-connected

model.add(keras.layers.Flatten())
# Appliquer le modèle sur chaque entrée
# As mentioned above, Siamese Network share weights between  tower networks (sister networks).
#To allow this, we will use same embedding network for both tower networks.
tower_1 = model(input_1) #siamese 1
tower_2 = model(input_2) #siamese 2

 # Concatenate the encoded inputs
    # Application de la fonction de distance L2 aux sorties de chaque branche
#concatenated = Lambda(euclidean_distance)([tower_1, tower_2])
concatenated = concatenate([tower_1, tower_2])

# Fully connected layers for prediction !!!!!!! fxer le nombre des noeuds
fc = Dense(50, activation='relu')(concatenated)
fc = Dense(50, activation='relu')(fc)
fc = Dense(50, activation='relu')(fc)
# Output layer for the affine transformation matrix
output = Dense(6, activation='linear')(fc)
    
siamese_net =keras.Model(inputs=[input_1,input_2],outputs=output)

#print(model.summary())
#print(siamese_net.summary())


# In[16]:


from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import regularizers
# Compiler le modèle
print("[INFO] compiling model...")
siamese_net.compile(loss='mean_squared_error', optimizer=Adam(), metrics=["accuracy"])
#loss=regularizers.l2(0.01)(' mean_absolute_error')
#siamese_net.compile(loss='mean_squared_error', optimizer='Adam')


# In[17]:


# Entraîner le modèle sur les données d'entrée
print("[INFO] training model...")
his = tf.keras.callbacks.History()
siamese_net.fit([data_left_train, data_right_train],
                affine_matrices_train , 
                batch_size=64,
                epochs=150,
                validation_data=([data_left_test, data_right_test], affine_matrices_test),
                callbacks=[his])


# In[18]:


'''
Les résultats que vous avez partagés montrent que la perte (loss)
diminue rapidement au cours des premières époques d'entraînement, 
puis atteint des valeurs très basses et reste à zéro pour les époques restantes.
Cela indique que votre modèle a surappris les données d'entraînement et a atteint une très bonne performance sur ces données,
mais cela ne garantit pas une bonne performance sur des données de test ou de validation.

Il est important de noter que pour éviter l'overfitting, il est recommandé 
de utiliser des techniques de régularisation comme la régularisation L1 ou L2, 
dropout, early stopping, etc. Il est également important de surveiller les
indicateurs de performance tels que la précision, la perte et la vitesse
d'entraînement sur des données de validation et non pas seulement sur les données d'entraînement pour évaluer
les performances de votre modèle.'''


# In[19]:


# Calculer la loss du modèle avec la base MPEG-7
Loss = siamese_net.evaluate([data_left_test, data_right_test], affine_matrices_test)
print("test loss", Loss)


# In[20]:


from sklearn.metrics import r2_score

#R-squared mesure la qualité de l'ajustement du modèle en indiquant la proportion 
#de la variance de la variable dépendante qui est expliquée par le modèle.


# make predictions on test data
predictions = siamese_net.predict([data_left_test, data_right_test])

# calculate R-squared
r2 = r2_score(affine_matrices_test, predictions)
print("R-squared:", r2)


# In[21]:


from sklearn.metrics import mean_squared_error
#MSE mesure également la précision des prévisions en indiquant la moyenne des erreurs 
#au carré entre les valeurs prédites et les valeurs réelles.
# Calculer la MSE entre les matrices prédites et réelles
mse = mean_squared_error(affine_matrices_test, predictions)
print("Mean Squared Error : ", mse)


# In[22]:


from sklearn.metrics import mean_absolute_error
#MAE mesure la précision des prévisions en indiquant la moyenne des erreurs absolues
#entre les valeurs prédites et les valeurs réelles.
# calculate mean absolute error
mae = mean_absolute_error(affine_matrices_test, predictions)
print("Mean Absolute Error:", mae)


# In[23]:


real_values = affine_matrices_test

distances = []
for i in range(len(predictions)):
    dist = np.linalg.norm(real_values[i]- predictions[i])
    distances.append(dist)

mean_distance = np.mean(distances)
print("Mean Euclidean Distance: ", mean_distance)


# In[24]:


# Prédire la matrice de transformation affine pour de nouvelles entrées
########predictions = siamese_net.predict([data_left_test, data_right_test])
# Afficher la matrice de transformation prédite
print(predictions.shape)
print(predictions[:,:])
predictions_Re=predictions.reshape(predictions.shape[0],2,3)
print(predictions_Re)


# In[25]:


print(predictions[101,:])
print(predictions[101,0])
print(predictions[101,1])
print(predictions[101,2])
print(predictions[101,3])
print(predictions[101,4])
print(predictions[101,5])


# In[26]:


print(siamese_net.optimizer.get_config())


# In[27]:


###TEST###
c=3
#Définir la matrice de transformation
transformation_matrix =predictions[c,:]
print(transformation_matrix)
print(affine_matrices_test[c,:])
#transformation_matrix = transformation_matrix.reshape(6, 1)
# Définir le point à transformer
 
x11=data_left_test[c,0]
x12=data_left_test[c,1]
print(x11)
x21=data_right_test[c,0]
x22=data_right_test[c,1]
Hx=x11*transformation_matrix[0]+x12*transformation_matrix[2]+transformation_matrix[4]
Hy=x11*transformation_matrix[1]+x12*transformation_matrix[3]+transformation_matrix[5]
print(Hx.shape)
print(Hy.shape)



# In[28]:


plt.plot(x11,x12,label=" original  shape")


# In[29]:


plt.plot(x21,x22,label="cible shape ")


# In[30]:



plt.plot(Hx,Hy,label=" transformed shape")


# In[31]:


plt.plot(x11,x12,label=" original  shape",c='r')
plt.plot(x21,x22,label="cible shape ",c='b')


# In[32]:


plt.plot(Hx,Hy,label=" transformed shape",c='r')
plt.plot(x21,x22,label="cible shape ",c='b')


# In[33]:


#Let's plot the curves for study
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[34]:


#Let's plot the curves for study
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('MPEG-7 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()


# In[ ]:




