import numpy as np
from flask import Flask, flash,  request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf 
#from efficientnet import EfficientNet
from keras.applications import MobileNet
from keras.models import load_model
import tensorflow_hub as hub
import os  
import cv2 
from PIL import Image

from flask import render_template

app = Flask(__name__)


MODEL_PATH = 'model_dense121v2.h5'
#model = load_model(MODEL_PATH,
       #custom_objects={'KerasLayer':hub.KerasLayer})
# USE WHEN WANT TO LOAD TRANSFER LEARNING 

model = load_model(MODEL_PATH)
from flask import render_template



@app.route('/')


@app.route('/home' )
def home():
    return render_template('home.html')



@app.route('/test_mri', methods=['GET'] )
def test_mri():
    return render_template('test_mri.html')

@app.route('/about' )
def about():
    return render_template('about.html')




@app.route('/predict', methods=['POST'])

def predict():
    try:
        imagefile= request.files['image']
        if imagefile.filename =='':
              return render_template('test_mri.html')
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'catch_image', secure_filename(imagefile.filename))
        
        imagefile.save(file_path)

        img =  cv2.imread(file_path)
        img = cv2.resize(img,(150,150))
        img_array = np.array(img)
        img_array = img_array.reshape(1,150,150,3)
        
 

        predictions = model.predict(img_array)
        labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
        
        prediction_class = labels[predictions.argmax()]
        
        
             
        prediction_probability = round (float(np.max(predictions)),5 ) 
      

        
        return render_template('result.html', prediction_class=prediction_class, prediction_probability=prediction_probability)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
