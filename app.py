import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import flask 
from flask import Flask
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text 

import source.config as config
from source.preprocessing import *
from source.transformer import *

app = Flask(__name__)

class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    (result,
     tokens,
     attention_weights) = self.translator(sentence, max_length=config.MAX_TOKENS)

    return result


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    response = translator(sentence).numpy()
    return flask.jsonify(response)

if __name__ == "__main__":
    translator = tf.saved_model.load('../translator')
    translator = ExportTranslator(translator)
    app.run(host="0.0.0.0", port="9999")