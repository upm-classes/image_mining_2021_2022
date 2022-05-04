### It requires the following packages
# These ones allows to create and activate the environment
# conda create --name image_mining python=3.8 -y
# conda activate image_mining
### These ones install the required libraries
### conda install -c conda-forge flask-restful
### conda install scikit-learn
### conda install pillow
### pip install tensorflow
### pip install scikit-image


from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from PIL import Image
import numpy as np
import io
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from skimage.transform import resize


app = Flask(__name__) # Server
api = Api(app) # api-rest

#  Loading the pre-trained model
model = VGG16(weights='imagenet', include_top=True)

def feature_extraction(image):
    # preprocessing
    image = resize(image, (224, 224), preserve_range=True,  anti_aliasing=True).astype(np.uint8)
    features = np.expand_dims(preprocess_input(image), axis=0)
    
    return features

# We create a resource for the api
class Prediction(Resource):
    @staticmethod
    def post():
        data = {"success": False}
        # Check if an image was posted
        if request.files.get('image'):
            im = request.files["image"].read()
            im = Image.open(io.BytesIO(im))
            im = np.array(im)

            features = feature_extraction(im)

            y_hat = model.predict(features)
            decoded = decode_predictions(y_hat)[0]

            prediction = {}
            for tuple in decoded:
                prediction[tuple[1]] = float(tuple[2])
            
            data['prediction'] = prediction
            data['success'] = True # Indicate that the request was a sucess

            return jsonify(data) # Response

api.add_resource(Prediction, '/predict')

if __name__ == "__main__":
    print('Loading model and Flask starting server...')
    print('please wait until server has fully started')

    app.run(debug=True, host='0.0.0.0', port=8888) # Debug mode and open to all connections in port 8888