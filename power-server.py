import flask as fl 
from flask import request, redirect, abort, jsonify
import numpy as np
#import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential

app = fl.Flask(__name__)


model = load_model('powerpred.h5')
#graph = tf.keras.models.get_default_graph()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/predict/<float:id>')
def predict(id):
    # Required because of a bug in Keras when using tensorflow graph cross threads
    #with graph.as_default():
    result = model.predict(np.array([[id]]))[0].tolist()
    data = {'Value': result}
    return fl.jsonify(data)






# Run in debug mode
if __name__ == "__main__":
   app.run(debug=True)