import os  
import cv2 
from keras.applications import MobileNet
from keras.models import load_model
import tensorflow_hub as hub
import os  
import cv2 
from PIL import Image
import numpy as np



imgShape = 150 
path= "E:\project -davv\TESTPROJECT\catch_image\gg_1.jpg"
img = cv2.imread(path)
img = cv2.resize(img,(150,150))
img_array = np.array(img)
img_array = img_array.reshape(1,150,150,3)




MODEL_PATH = 'model_dense121v2.h5'
#model = load_model(MODEL_PATH,
       #custom_objects={'KerasLayer':hub.KerasLayer})

model = load_model(MODEL_PATH)
predictions = model.predict(img_array)
print ( predictions)
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
        
prediction_class = labels[predictions.argmax()]
print ( prediction_class)
