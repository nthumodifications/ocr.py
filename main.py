from time import time
from fastapi import FastAPI
from keras.models import load_model
import keras.utils as image
import numpy as np
import requests
import cv2

app = FastAPI()
model = load_model('model.h5')

def load(url: str):
  response = requests.get(url)
  content = response.content
  with open('test.png', 'wb') as f:
    f.write(content)

def pad(image, size = (30, 30)):
  h, w = image.shape[:2]
  dh, dw = size[0] - h, size[1] - w
  top, left = dh // 2, dw // 2
  bottom, right = dh - top, dw - left
  padding = ((top, bottom), (left, right)) + ((0, 0),) * (image.ndim - 2)
  return np.pad(image, padding, 'constant')

def process(file: str):
  image = cv2.imread(file, 0)
  _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
  subimages = []
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    subimage = image[y:y+h, x:x+w]
    subimages.append(subimage)
  return subimages


@app.get('/')
async def uwu(url: str):
  load(url)
  images = process('test.png')

  answers = []
  for img in images:
    img = pad(img) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    prediction = np.argmax(predictions)
    answers.append(prediction)
  
  answer = int(''.join(map(str, answers)))
  return answer