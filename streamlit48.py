import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
model = tf.keras.models.load_model("my_h5_model6 (1).h5")
original_title = '<p style="font-family:Tahoma; color:White; font-size: 32px; text-align: center;">Welcome to COVID-19 Predictor</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.image("bg.jpg",width=700)
uploaded_file = st.file_uploader("Please upload your chest radiograph(jpg file only)",type="jpg")
map_dict = {0: 'COVID19',
            1: 'Normal',
            2: 'Pneumonia'}
if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes,1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  resized = cv2.resize(opencv_image,(224,224))
  st.image(opencv_image, channels="RGB")
  resized = mobilenet_v2_preprocess_input(resized)
  img_reshape = resized[np.newaxis,...]
   
  Generate_pred = st.button("Generate Prediction")
  if Generate_pred:
    prediction = model.predict(img_reshape).argmax()
    st.title("Predicted label on image is {}".format(map_dict[prediction]))