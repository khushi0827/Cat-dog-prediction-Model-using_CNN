# Liberaries
import os 
import numpy as np
import cv2
from flask import Flask , render_template,request
from  tensorflow.keras.models import load_model

# creating app

app = Flask(__name__)

# path to save image files which will be uploaded by user
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

# getting model from directory
model = load_model('model.h5')

# classes labels
Classes = ['Cat','Dog']

def preprocess_image(filepath):
    img=cv2.imread(filepath) # Read

    img=cv2.resize(img,(256,256)) # resize [256,256]

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # convert BGR to RGB
 
    img=img/255 # Normalize between 0 to 1

    img=np.reshape(img,(1,256,256,3)) # reshape as per training data
   
    return img # getting image array out from function

@app.route('/')
def index():
    return render_template('index.html') # connecting home page as index file

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename=='':
        return 'No file selected'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filepath)
    
    img_array = preprocess_image(filepath)
   
    prediction = model.predict(img_array)
   
    result = Classes[int(prediction[0][0])>0.5]

    return render_template('result.html',prediction=result,img_path = filepath)

if __name__ == '__main__':
    app.run(debug=True)


    
