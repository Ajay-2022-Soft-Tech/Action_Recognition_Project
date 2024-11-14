#!/usr/bin/env python
# coding: utf-8

# ## What is Human Action Recognition(HAR)?
# 
# - Human activity recognition, or HAR for short, is a broad field of study concerned with identifying the specific movement or action of a person based on sensor data.
# - Movements are often typical activities performed indoors, such as walking, talking, standing, and sitting

# ## Why it is important ?
# - Human activity recognition plays a significant role in human-to-human interaction and interpersonal relations.
# - Because it provides information about the identity of a person, their personality, and psychological state, it is difficult to extract.
# - The human ability to recognize another person’s activities is one of the main subjects of study of the scientific areas of computer vision and machine learning. As a result of this research, many applications, including video surveillance systems, human-computer interaction, and robotics for human behavior characterization, require a multiple activity recognition system.

# ## What is a CNN?
# 
# A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
# 
# CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing (NLP).
# 
# ![cnn](./images/cnn.jpeg)
# 
# ## VGG16
# 
# VGG-16 is a convolutional neural network that is 16 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.
# 
# ![vgg16](./images/vgg16.png)

# ## What is Transfer Learning
# 
# Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.
# 
# ![transfer learning](./images/transfer.jpeg)

# ## HAR using Transfer Learning

# ## Imports

# In[1]:


import os
import glob

#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Scikit Learn
from sklearn.model_selection import train_test_split

#Image processing
import cv2 as cv

# #tensorflow 
# from tensorflow.keras.utils import to_categorical
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
# from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#Dataframe classifying the actions
train_action = pd.read_csv("data/Training_set.csv")
test_action = pd.read_csv("data/Testing_set.csv")


# In[3]:


train_action.head()


# ## Looking at the data

# In[4]:


#Printing the images along with their respective action

img = cv.imread('data/train/' + train_action.filename[0])
plt.title(train_action.label[0])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB));


# In[5]:


test_action.shape


# There are 12,600 images in the training dataset. Creating a function which will randomly sample a image from the set and print the image along with its labeled action.

# In[6]:


#Sample images and their labels in the training data


def show_img_train():
    img_num = np.random.randint(0,12599)
    img = cv.imread('data/train/' + train_action.filename[img_num])
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(train_action.label[img_num])

def show_img_test():
    img_num = np.random.randint(0,5399)
    img = cv.imread('data/test/' + test_action.filename[img_num])
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


# In[7]:


show_img_train()


# In[8]:


show_img_train()


# In[9]:


show_img_train()


# In[10]:


show_img_train()


# In[11]:


show_img_train()


# In[12]:


show_img_test()


# In[13]:


show_img_test()


# Note that the images from the test set do not contain the labels. That is for the model to predict.

# ## Plotting the label classes

# ### Pie chart

# In[14]:


l = train_action.label.value_counts()
fig = px.pie(train_action, values=l.values, names=l.index, title='Distribution of Human Activity')
fig.show()


# ### Value counts

# In[15]:


train_action.label.value_counts()


# The output classes for the image classification are balanced. This reduces a step in pre-processing which deals with imbalanced classes using techniques such as SMOTE, under-sampling, etc.

# ## Preprocessing

# In[16]:


img = cv.imread('data/train/' + train_action.filename[0])

#OpenCV reads the images in BGR instead of the standard RGB, hence the below line of code

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB));
print(img.shape)


# The inputs for VGG16 model as per the Tensorflow documentation is 224x224. Here out training set images come in all different sizes. We need to resize the images.

# ### Create the directory for resized images 

# Skipping the below cell since the folder has already been created.

# In[17]:


#create resized directory in the current project folder
parent_dir = os.getcwd()
directory = 'resized-train'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
print('Created "resized-train" folder')


# ## Resize the training images

# In[18]:


#dimensions
width = 224
height = 224
dim = (width, height)

#resizing all the images in the train folder
for i in np.arange(len(train_action.filename)):    
    #read the filename from the dataframe
    filename = train_action.filename.iloc[i]
    #read the image from the train folder
    img = cv.imread('data/train/' + filename)
    #resize the image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    #write the image in resized folder
    cv.imwrite('./resized-train/' + filename ,resized)


# ## Resize the test set images

# In[19]:


#create resized directory in the current project folder
parent_dir = os.getcwd()
directory = 'resized-test'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
print('Created "resized-test" folder')


# In[20]:


#resizing all the images in the train folder
for i in np.arange(len(test_action.filename)):    
    #read the filename from the dataframe
    filename = test_action.filename.iloc[i]
    #read the image from the train folder
    img = cv.imread('data/test/' + filename)
    #resize the image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    #write the image in resized folder
    cv.imwrite('./resized-test/' + filename ,resized)


# ## Pre-processing

# ### Display image

# In[21]:


img = cv.imread('resized-train/' + train_action.filename[0])
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
print(img.shape)


# We can see that the images have been resized and stored in a different directory. This helps us avoid resizing the images everytime we run the notebook.

# ## Read images and convert them into numpy arrays

# ## X

# In[22]:


#empty list train
X = []

#reading all the resized images
for i in np.arange(len(train_action.filename)): 
    img = cv.imread('resized-train/' + train_action.filename[i])
    X.append(img)

X = np.asarray(X)
X.shape


# ## y

# In[23]:


y = np.asarray(pd.get_dummies(train_action.label))
print(y.shape)


# ## Test train Split

# In[24]:


print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42,stratify=train_action.label)


# In[25]:


print(X_train.shape,y_train.shape)


# ## Unseen test set
# 
# This set of images are meant to be submitted on Kaggle. We wont be using this data for this project.

# In[26]:


#empty list test
X_unseen = []

#reading all the resized images
for i in np.arange(len(test_action.filename)): 
    img = cv.imread('resized-test/' + test_action.filename[i])
    X_unseen.append(img)

X_unseen = np.asarray(X_unseen)
X_unseen.shape


# ## Building the CNN model using transfer learning

# ### Initializing a VGG16 model

# In[27]:


from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
# from keras.preprocessing.image import ImageDataGenerator
#Sequential model constructor
cnn_model = Sequential()

#initializing a vgg16 wihtout the top layers 
pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

#Freezing the deeper layers
for layer in pretrained_model.layers:
        layer.trainable=False
        
#adding our layers to the model
cnn_model.add(pretrained_model)
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(15, activation='softmax'))


# ## Model Summary

# - The loss function that we are trying to minimize is Categorical Cross Entropy. This metric is used in multiclass classification. This is used alongside softmax activation function.
#  
# - Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. This algorithm is straight forward to implement and computationally efficient.

# In[28]:


cnn_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
cnn_model.summary()


# ## Training the model

# In[29]:


history = cnn_model.fit(X_train,y_train, epochs=4)


# ## Save the model weights

# In[ ]:


# Saving the trained weights with the required filename format
cnn_model.save_weights('./model/weights.weights.h5')


# In[ ]:


#plotting the loss
loss = history.history['loss']

plt.title("Loss Function")
plt.xlabel("epochs")
sns.set_theme(style='darkgrid')
sns.lineplot(loss)


# In[ ]:


plt.title("Accuracy")
plt.xlabel("epochs")
accu = history.history['accuracy']
sns.lineplot(accu)


# ## Predictions

# In[ ]:


y_preds = cnn_model.predict(X_test)


# ## Accuracy and Log Loss

# In[ ]:


from sklearn.metrics import accuracy_score, log_loss

print('Log Loss:',log_loss(np.round(y_preds),y_test))
print('Accuracy:',accuracy_score(np.round(y_preds),y_test))


# ## Visualizing the predictions

# In[ ]:


from PIL import Image
import matplotlib.image as mimg

#reading image and resize 
def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((224,224)))

#predict the class and the confidence of the prediction
def test_predict(test_image):
    result = cnn_model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)

    image = mimg.imread(test_image)
    plt.imshow(image)
    plt.title(prediction)


# In[ ]:


#Image 1
filename = test_action.filename.iloc[200]
test_predict('./resized-test/' + filename)


# In[ ]:


#Image 2
filename = test_action.filename.iloc[10]
test_predict('./resized-test/' + filename)


# ## Next Steps and Recommendations
# 
# 
# - In order to improve the accuracy, we can unfreeze few more layers and retrain the model. This will help us further improve the model.
# 
# - We can tune the parameters using KerasTuner.
# 
# - The model reached a good accuracy score after the 20 epochs but it has been trained for 60 epochs. There is definitely some overfitting which can avoided with early stopping.
# 
# - The nodes in the deep layers were connected. We can introduce some amount dropout for regularization.
