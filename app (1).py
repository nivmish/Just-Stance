import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from sklearn.utils import shuffle
import tensorflow.keras as keras
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import av
import queue
import time
import pandas as pd

class VideoProcessor: #class for processing inputed frames
    result_queue: "queue.Queue[List[Detection]]"
    def __init__(self) -> None:
        self.result_queue = queue.Queue() #queue for saving results of frame

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24") #load image from camera
        #first model
        #Transformations to fit first model
        x = cv2.resize(image, (512, 512)) #resizing
        x = x/255.0 #dividing by 255
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0) #turning to a vector
        #Segmenting
        y = model_segment.predict(x)[0] #making mask
        y = cv2.resize(y, (size, size)) #returning to original size
        y = np.expand_dims(y, axis=-1)
        sil = np.full((size,size,3), 255) #creating a white blnank image
        image = cv2.resize(image, (size, size)) #reshaping original frame
        masked_image = sil * y #putting the mask over the blank image, making a white silhouette
        #making the frame to return
        for i in range(size): #going over each pixel of the original frame and the silhouette  
          for j in range(size):
              check = masked_image[i,j] #pixel to check
              change = image[i,j] #pixel to change
              if ((check[0] > 250) and (check[1] > 250 ) and (check[2] > 250)):#if check pixel is white
                #change color of change pixel to green
                change[0] = 4
                change[1] = 244
                change[2] = 4
        #seconed model
        #Transformations to fit second model
        masked_image[::] = 255 - masked_image[::] #flip colors  
        to_send =  masked_image.astype(np.uint8) 
        letter = cv2.resize(masked_image, (32, 32)) #resizing        
        test_image = tf.keras.preprocessing.image.img_to_array(letter) #change to vector
        test_image = np.expand_dims(test_image, axis = 0)
        #classification
        result = model_letters.predict(test_image)
        self.result_queue.put(get_result(result))#save results of predicted letter in the queue
        return av.VideoFrame.from_ndarray(image.astype(np.uint8), format="bgr24")#return frame to show


def get_result(result):
    index = np.argmax(result)#array of chances for each letter
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",][index] #return the letter with the highest chance


def iou(y_true, y_pred):#calculates iou of prediction
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum() #create intersection
        union = y_true.sum() + y_pred.sum() - intersection #create union
        x = (intersection + 1e-15) / (union + 1e-15) #intersection over union
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):#calculates dice coeffient of prediction
    y_true = tf.keras.layers.Flatten()(y_true) #transform to vector
    y_pred = tf.keras.layers.Flatten()(y_pred) #transform to vector
    intersection = tf.reduce_sum(y_true * y_pred) #create union
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth) #2 * |y_true ∩ y_pred| / (|y_true| + |y_pred|)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)#loss is 1 -accuracy



 #Loading models
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}): #custom loss function
  model_segment = tf.keras.models.load_model("/content/drive/MyDrive/AIproject/model.h5")
model_letters = keras.models.load_model('/content/drive/MyDrive/AIproject/model_letters')


st.title("JUST STANCE")#put title
 
size = 400 #frame size
webrtc_ctx = webrtc_streamer( #put a live webcam fid widget
        key="opencv-filter",
        video_processor_factory = VideoProcessor,#custom function on each frame
        media_stream_constraints = { #specs of input(size and type)
    "audio": False,
    "video": {
        "width": {
            "min": size,
            "max": size
        },
        "height": {
            "min": size,
            "max": size
        }
    }
},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})#server configuration


letters = shuffle(['Y','F','I','X'])#letters needed to preform
index = 0 #index which letter is played
score = [[0]] #score keeper
current_let = 'A' #defult start value for predictions
if webrtc_ctx.state.playing: #if live webcam widget is on
    instruct = st.header('Make the letter with your body: '+ letters[index]) #write instructions
    show_current = st.empty()#text widget
    sitch = st.empty()#text widget of progress
    while True: #loop of receving frames, getting results and returning alterd frames 
      if webrtc_ctx.video_processor: #if data is given from widget 
        try:
          score[0][index] += 1 #add 1 to that letter's score
          result = webrtc_ctx.video_processor.result_queue.get(
            timeout=1.0 #receve frame and process
          )  
        except queue.Empty:
          result = None
        if result != None and result != current_let:#if frame was receved and the letter in it is not the same as what was previously perdicted
          current_let = result #change the letter on screen which shows the current guess of the game
          if current_let == letters[index]: #if the prediction is corecct 
            index += 1 #go to next letter        
            if index == len(letters): #if all letters were done
              st.balloons() #show balloons
              sitch.write("Thanks for playing") #text widget
              df = pd.DataFrame( #show score in table format
              score,
              columns=('Letter %d' % i for i in range(1,len(letters)+1)))
              st.write("Your score is: " + str(sum(score[0])))
              st.table(df)
              time.sleep(300)
            else: #if there are more letters to do
              sitch.write("Good Job!!! " + str(len(letters) - index) + " letters to go.")   #text image indicating how many letters are left
              instruct.header('Make the letter with your body: '+ letters[index]) #write instructions
              score[0].append(0) #make score for new letter
        show_current.write("Letter detected is: " + current_let) #show current predicted letter      
      else:
        break