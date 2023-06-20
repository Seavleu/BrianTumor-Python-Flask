import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tfe
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import normalize 
from keras.models import Sequential


no_tumor_images=os.listdir('../brain-tumor-detection-master/brain_tumor/Training/no_tumor/')
have_tumor_images=os.listdir('../brain-tumor-detection-master/brain_tumor/Training/pituitary_tumor/')
dataset=[]
label=[]
INPUT_SIZE=64

# no tumor 
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread('../brain-tumor-detection-master/brain_tumor/Training/no_tumor/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

# have tumor
for i, image_name in enumerate(have_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread('../brain-tumor-detection-master/brain_tumor/Training/pituitary_tumor/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

print(len(dataset))
print(len(label))

dataset=np.array(dataset)
label=np.array(label)

# Train  80% , Test 20% and Split

x_train, x_test, y_train, y_test= train_test_split(dataset,label,test_size=0.2, random_state=0)
# Reshape = (n, image_width, image_height, n_channel)

print("Train  80%: ",x_train.shape)
print(y_train.shape)

print("Test  20%: ",x_test.shape)
print(y_test.shape)

# 3 is channel RGB

# Categorise
x_train = normalize(x_train,axis=1)
x_test = normalize(x_train,axis=1)