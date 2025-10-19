import urllib.request
import os

# Pre-trained emotion detection model
model_url = "https://github.com/serengil/tensorflow-101/raw/master/model/facial_expression_model_weights.h5"

# Downloading the model
print("Downloading pre-trained emotion detection model...")
urllib.request.urlretrieve(model_url, "emotion_model.h5")

print("Model downloaded successfully!")
