#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It is preventable and curable.
# 
# In 2017, there were an estimated 219 million cases of malaria in 90 countries.
# Malaria deaths reached 435 000 in 2017.
# The WHO African Region carries a disproportionately high share of the global malaria burden. In 2017, the region was home to 92% of malaria cases and 93% of malaria deaths.
# Malaria is caused by Plasmodium parasites. The parasites are spread to people through the bites of infected female Anopheles mosquitoes, called "malaria vectors." There are 5 parasite species that cause malaria in humans, and 2 of these species – P. falciparum and P. vivax – pose the greatest threat.
# 
# # Challenges with Diagnosis 
# 
# Where malaria is not endemic any more (such as in the United States), health-care providers may not be familiar with the disease. Clinicians seeing a malaria patient may forget to consider malaria among the potential diagnoses and not order the needed diagnostic tests. Laboratorians may lack experience with malaria and fail to detect parasites when examining blood smears under the microscope.
# Malaria is an acute febrile illness. In a non-immune individual, symptoms usually appear 10–15 days after the infective mosquito bite. The first symptoms – fever, headache, and chills – may be mild and difficult to recognize as malaria. If not treated within 24 hours, P. falciparum malaria can progress to severe illness, often leading to death.
# Microscopic Diagnosis
# 
# Malaria parasites can be identified by examining under the microscope a drop of the patient’s blood, spread out as a “blood smear” on a microscope slide. Prior to examination, the specimen is stained to give the parasites a distinctive appearance. This technique remains the gold standard for laboratory confirmation of malaria. However, it depends on the quality of the reagents, of the microscope, and on the experience of the laboratorian.
# 
# # Steps to solve the problem :-
# 
# * Importing Libraries.
# * Loading the data.
# * Data preprocessing.
# * Data augmentation.
# * Ploting images and its labels to understand how does an infected cell and uninfected cell looks like.
# * Spliting data in Train , Evaluation and Test set.
# * Creating a Convolution Neural Network function.
# * Wrapping it with Tensorflow Estimator function.
# * Training the data on Train data.
# * Evaluating on evaluation data.
# * Predicting on Test data
# * Ploting the predicted image and its respective True value and predicted value.

# In[1]:


# Importing the relevant libraries


# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from imutils import paths
import argparse
import skimage


# In[2]:


# reading in the dataset

dataset = r'C:\Users\animu\Downloads\malaria'

# creating a dictionary to store and iterate through the dataset
args = {}
args['dataset'] = dataset


# separating the data features from the labels and storing them in lists



ipaths = list(paths.list_images(args['dataset']))
features = []
labels = []
for i in ipaths:
    label = i.split(os.path.sep)[-2]
    image = cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    labels.append(label)
    features.append(image)

data = np.array(features)/255.0
labels = np.array(labels)


# In[3]:


# Visualizing the data

infected_images = os.listdir(dataset + '/Malaria/')
normal_images = os.listdir(dataset + '/Normal/')

def cell_image_plotter(i):
    uninfected = cv2.imread(dataset + '//Normal//' + normal_images[i])
    uninfected = skimage.transform.resize(uninfected, (150,150,3))
    malaria  = cv2.imread(dataset + '//Malaria//' + infected_images[i])
    malaria = skimage.transform.resize(malaria, (150,150,3), mode = 'reflect')
    paired = np.concatenate((malaria,uninfected), axis = 1)
    print('Malaria Parasitized vs Uninfected Red Blood Cell')
    plt.figure(figsize = (10,5))
    plt.imshow(paired)
    plt.show()
    
for i in range(5):
    cell_image_plotter(i)
    
'''The Malaria infected cells on the left can be clearly distinguished by the granulation or small dot present within them'''    


# In[4]:


# Transforming the labels to categorical values

binarizer = LabelBinarizer()
labels = binarizer.fit_transform(labels)
labels = to_categorical(labels)
labels


# In[5]:


# Now that the features and labels are stored in the appropriate format, we can split our data for training

X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                    random_state = 7,
                                                    shuffle =True, 
                                                    stratify = labels,
                                                    test_size = .2)

# creating more images using image augmentation

training_data_aug = ImageDataGenerator(rotation_range=20,
                                       zoom_range=0.15,
                                        width_shift_range=0.2, 
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                    horizontal_flip=True,
                                       fill_mode="nearest"
                                      )


# In[6]:


# creating the C N N model

base_model = VGG16(include_top= False,
                        weights = 'imagenet',
                         input_tensor= Input(shape=(224,224,3))
                         )

base_model.summary()


# In[7]:


# removing the pretrained output layers and substituting it with our desired binary output layer

head_model = base_model.output
head_model = AveragePooling2D(pool_size = (4,4))(head_model)
head_model = Flatten(name = 'flatten')(head_model)
head_model = Dense(64, activation = 'sigmoid')(head_model)
head_model = Dropout(0.5)(head_model) # dropout layer prevents overfitting
head_model = Dense(2, activation = 'softmax')(head_model) # our output layer containing our binary category

model = Model(base_model.input, head_model)

for layer in base_model.layers:
    layer.trainable = False


# In[8]:


model.summary()


# In[ ]:


# compiling the model

learning_rate = 1e-3
epochs = 10
batch_sizes = 32
opt = Adam(lr = learning_rate, 
                 decay = learning_rate//epochs)

model.compile(loss= 'binary_crossentropy', optimizer = opt, metrics= ['accuracy'])

# fitting the model with the augmented data

generator = model.fit(
                    training_data_aug.flow(X_train, y_train, batch_size= batch_sizes),
                               steps_per_epoch = len(X_train)//batch_sizes,
                                validation_data = (X_test, y_test),
                               validation_steps= len(X_test)//batch_sizes,
                               epochs = epochs
                    )
                               


# In[ ]:


# Visualizing the test predictions
length = 4
width = 5

fig, ax = plt.subplots(length,width, figsize = (13,13))
ax = ax.ravel()
pred = model.predict(X_test, batch_size = batch_sizes)
for i in np.arange(0,length*width):
    ax[i].imshow(X_test[i])
    ax[i].set_title('Prediction = {}\n True = {}'.format(pred.argmax(axis =1)[i], y_test.argmax(axis =1)[i]))
    ax[i].axis('off')
plt.subplots_adjust(wspace = 1, hspace =1)    


# In[ ]:


#calculating the prediction accuracy and printing the classification report

y_prediction = model.predict(X_test, batch_size = batch_sizes)
y_prediction = np.argmax(y_prediction, axis = 1)
print(classification_report(y_test.argmax(axis = 1),
                            y_prediction, target_names = binarizer.classes_))

print(accuracy_score(y_test.argmax(axis=1), y_prediction)*100 )


# In[ ]:


# plotting the training and validation loss and accuracy

# plotting loss
plt.plot(generator.history['loss'], label = 'Training Loss')
plt.plot(generator.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.show()
plt.savefig('training_validation_loss')

# plotting accuracy
plt.plot(generator.history['accuracy'], label = 'Training Accuracy')
plt.plot(generator.history['val_accuracy'], label = 'Validation Accuracy')
plt.legend()
plt.show()


# In[ ]:


# saving the model

model.save(r'C:\Users\animu\Downloads\malaria\malaria_classifier.v3')


# In[ ]:


# Testing the model

# mal_model = load_model(r'C:\Users\animu\Downloads\malaria\malaria_classifier.v3')

# random image from dataset
img1 = image.load_img(r'C:\Users\animu\Downloads\malaria\Normal\C13NThinF_IMG_20150614_131417_cell_110.png',
                     target_size=(224, 224)
                    )


imgplot = plt.imshow(img1)
x = image.img_to_array(img1)
x = np.expand_dims(x, axis = 0)
image_data = preprocess_input(x)
classes = mal_model.predict(image_data)
new_pred = np.argmax(classes, axis = 1)

if new_pred == [1]:
    print('Prediction: Normal')
elif new_pred == [0]:
    print("Prediction: Malaria")

    


# In[ ]:


img2 = image.load_img(r'C:\Users\animu\Downloads\malaria\Malaria\C48P9thinF_IMG_20150721_160944_cell_217.png',
                     target_size=(224, 224)
                    )
imgplot2 = plt.imshow(img2)
x2 = image.img_to_array(img2)
x2 = np.expand_dims(x2, axis = 0)
image_data2 = preprocess_input(x2)
classes2 = mal_model.predict(image_data2)
new_pred2 = np.argmax(classes2, axis = 1)

if new_pred2 == [1]:
    print('Prediction: Normal')
else:
    print("Prediction: Malaria")

