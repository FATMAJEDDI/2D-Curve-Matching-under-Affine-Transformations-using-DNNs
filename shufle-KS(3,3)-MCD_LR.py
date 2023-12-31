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


dataframe = pd.read_csv('database_CMD_labled5.csv',header = None)
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
print(data_left_train.shape)
data_right_train = Reshape_PostCon[:train_size]
print(data_right_train.shape)
affine_matrices_train = Transformation_matrix[:train_size]
print(affine_matrices_train.shape)


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
model.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu',  padding='same', input_shape=shape, name="Conv2D_1"))
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',  padding='same', name="Conv2D_2"))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',  padding='same', name="Conv2D_3"))
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


print("[INFO] compiling model...")
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.model_selection import learning_curve
# Entraîner le modèle sur les données d'entrée
print("[INFO] training model...")
optimizer = keras.optimizers.Adam()
learning_rates = [0.001, 0.01, 0.1, 1.0] # Define the learning rates to evaluate

for lr in learning_rates:
    # Set the learning rate for the optimizer
    optimizer.lr.assign(lr)
    # Train the model
    siamese_net.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae','accuracy'])
    # Concaténer les données d'entrée
    X_train = np.concatenate((data_left_train, data_right_train), axis=1)
    # Calculate the learning curve
    print("[INFO] Calculating learning curve...")
    train_sizes, train_scores, valid_scores = learning_curve(siamese_net, X_train, affine_matrices_train,
                                                            cv=5, scoring='neg_mean_squared_error')

    # Calculate the mean MSE and MAE scores
    mean_mse = -np.mean(valid_scores, axis=1)
    mean_mae = -np.mean(valid_scores, axis=1)

    # Store the values in the corresponding lists
    mse_values.append(mean_mse)
    mae_values.append(mean_mae)

# Plot the learning curves
plt.figure()
plt.title("Learning Curves with Varying Learning Rates")
plt.xlabel("Training Examples")
plt.ylabel("Mean Squared Error")
plt.grid()

for i, lr in enumerate(learning_rates):
    plt.plot(train_sizes, mse_values[i], label="Learning Rate: " + str(lr))

plt.legend(loc="best")
plt.show()
   
  



# Afficher les résultats
print("Train sizes:", train_sizes)
print("Mean MSE scores:", mean_mse)
print("Mean MAE scores:", mean_mae)


# In[ ]:


# Evaluate the model
# Évaluation du modèle sur les données de test
loss= siamese_net.evaluate([data_left_test, data_right_test], affine_matrices_test)




# In[ ]:


learning_rate_index = learning_rates.index(0.001)
mse_value = mse_values[learning_rate_index]
mae_value = mae_values[learning_rate_index]

print(f"MSE for learning rate 0.001: {mse_value}")
print(f"MAE for learning rate 0.001: {mae_value}")


# In[ ]:


# Récupération des historiques de MSE et MAE
mse_history = history.history['mse']
mae_history = history.history['mae']

  # Affichage des courbes de MSE et MAE
plt.plot(mse_history, label=f'LR={lr}')
plt.plot(mae_history, label=f'LR={lr}')

# Configuration du graphique
plt.title('MSE and MAE vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Affichage du graphique
plt.show()


# In[ ]:


predictions = model.predict([data_left_test, data_right_test])
mae = tf.keras.metrics.mean_absolute_error(affine_matrices_test, predictions).numpy()
mae_results.append(mae)
print(f"Learning rate: {lr} - Test loss: {loss} - MAE: {mae}")


# In[ ]:


plt.plot(learning_rates, mse_results, label='MSE')
plt.plot(learning_rates, mae_results, label='MAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import r2_score

#R-squared mesure la qualité de l'ajustement du modèle en indiquant la proportion 
#de la variance de la variable dépendante qui est expliquée par le modèle.


# make predictions on test data
predictions = siamese_net.predict([data_left_test, data_right_test])

# calculate R-squared
r2 = r2_score(affine_matrices_test, predictions)
print("R-squared:", r2)


# In[ ]:


from sklearn.metrics import mean_squared_error
#MSE mesure également la précision des prévisions en indiquant la moyenne des erreurs 
#au carré entre les valeurs prédites et les valeurs réelles.
# Calculer la MSE entre les matrices prédites et réelles
mse = mean_squared_error(affine_matrices_test, predictions)
print("Mean Squared Error : ", mse)


# In[ ]:


from sklearn.metrics import mean_absolute_error
#MAE mesure la précision des prévisions en indiquant la moyenne des erreurs absolues
#entre les valeurs prédites et les valeurs réelles.
# calculate mean absolute error
mae = mean_absolute_error(affine_matrices_test, predictions)
print("Mean Absolute Error:", mae)


# In[ ]:


real_values = affine_matrices_test

distances = []
for i in range(len(predictions)):
    dist = np.linalg.norm(real_values[i]- predictions[i])
    distances.append(dist)

mean_distance = np.mean(distances)
print("Mean Euclidean Distance: ", mean_distance)


# In[ ]:


# Prédire la matrice de transformation affine pour de nouvelles entrées
########predictions = siamese_net.predict([data_left_test, data_right_test])
# Afficher la matrice de transformation prédite
print(predictions.shape)
print(predictions[:,:])
predictions_Re=predictions.reshape(predictions.shape[0],2,3)
print(predictions_Re)


# In[ ]:


print(predictions[101,:])
print(predictions[101,0])
print(predictions[101,1])
print(predictions[101,2])
print(predictions[101,3])
print(predictions[101,4])
print(predictions[101,5])


# In[ ]:


print(siamese_net.optimizer.get_config())


# In[ ]:


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



# In[ ]:


plt.plot(x11,x12,label=" original  shape")


# In[ ]:


plt.plot(x21,x22,label="cible shape ")


# In[ ]:



plt.plot(Hx,Hy,label=" transformed shape")


# In[ ]:


plt.plot(x11,x12,label=" original  shape",c='r')
plt.plot(x21,x22,label="cible shape ",c='b')


# In[ ]:


plt.plot(Hx,Hy,label=" transformed shape",c='r')
plt.plot(x21,x22,label="cible shape ",c='b')


# In[ ]:


#Let's plot the curves for study
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


#Let's plot the curves for study
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()


# In[ ]:




