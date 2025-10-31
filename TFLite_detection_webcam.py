######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras, Tim West
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import serial
from threading import Thread
import importlib.util
import random
from playsound import playsound
#import csv
#speak
from gtts import gTTS
import os

debug_tw = 0
personlosttime = 2

class speakToMe:
    
    bequietuntil = 0
  
    def speak(self, text):
        ts = time.time()
      
        if (ts > self.bequietuntil):
            tts = gTTS(text, lang='en')
            tts.save('hello.mp3')
            file_size = os.path.getsize('hello.mp3')/1024
            #print("File Size is :", file_size, " kb")
            os.system('mpg321 hello.mp3 -q &')
            #playsound("hello.mp3", block = False)
            self.bequietuntil = ts + file_size/4
            #print(text + ", ts = " + str(ts) + ", quietuntil = " + str(self.bequietuntil))
            

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera or USB camera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


def picture_in_picture(main_image, overlay, x_offset, y_offset):
    """
    Overlay an image onto a main image with a white border.
    
    Args:
        main_image (image): The main image.
        overlay_image (image): The overlay image.
        img_ratio (int): The ratio to resize the overlay image height relative to the main image.
        border_size (int): Thickness of the white border around the overlay image.
        x_margin (int): Margin from the right edge of the main image.
        y_offset_adjust (int): Adjustment for vertical offset.

    Returns:
        np.ndarray: The resulting image with the overlay applied.
    """
    
    y_offset = y_offset + 20
    x_offset = x_offset - 338

    if x_offset < 0:
	    return main_image

    
    alpha_channel = overlay[:, :, 3] / 255.0
    overlay_colors = overlay[:, :, :3]
    
    alpha_mask_3ch = np.stack([alpha_channel, alpha_channel, alpha_channel], axis = 2)
    # ghostly alpha
    alpha_mask_3ch = 0.5 * alpha_mask_3ch
    
    
    original = main_image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]]
    
    mixedimage = (overlay_colors * alpha_mask_3ch + original * (1-alpha_mask_3ch)).astype(np.uint8)
    

    # Overlay the image
    main_image[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = mixedimage #overlay_with_border

    return main_image


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.60)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Read costumes, greetings, and prompts
with open("Costumes.txt") as f:
    costumelist = f.read().splitlines()
#print(costumelist)

with open("greeting.txt") as f:
    greetinglist = f.read().splitlines()

with open("prompt.txt") as f:
    promptlist = f.read().splitlines()


if debug_tw == 1:
    print(costumelist)
    print(greetinglist)
    print(promptlist)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Initialize video save
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filename = "/home/orangepi/tflite1/Videos/video" + str(round(time.time())) + ".mp4"
print("Filename: " + filename)
out = cv2.VideoWriter(filename, fourcc, 20.0, (imW, imH))

# Initialize chatterbox
chatterbox = speakToMe()



# Use this if we are doing the head with eyes moving.
# Setup Serial
use_serial = 0 # 1 = on, 0 = off

# Eye degree
degree = 90
eyeslastupdated = time.time()
eyesopen = 0 # 1 = open, 0 = closed

if (use_serial == 1):
	#print(serial.tools.list_ports())
	SerialObj = serial.Serial('/dev/ttyUSB0')                                                     
	SerialObj.baudrate = 9600  # set Baud rate to 9600
	SerialObj.bytesize = 8   # Number of data bits = 8
	SerialObj.parity  ='N'   # No parity
	SerialObj.stopbits = 1   # Number of Stop bits = 1



# Load an image of a ghost to display
# Load an image
ghostimage = cv2.imread('images/ghost.png', cv2.IMREAD_UNCHANGED)  

if ghostimage is None:
  print("Error: Could not load image.")
else:
  print("Loaded the ghost image.")

# Position of the ghost image:
ghostpositionx = 0
ghostpositiony = 0


# Setup for running
fontsize = 2
primary_person_found = 0
personlastseentime = time.time()
personlastseentickcount = 0

costume = random.choice(costumelist)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    if frame1 is None:
        break
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold

    # This keeps track of person found in last image
    primary_person_found_in_last_image = primary_person_found
    
    # This keeps track of the parimary person in this image
    primary_person_found = 0


    for i in range(len(scores)):
    
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            # Red color for non-primary person
            color = (0, 10, 255)

            # Label for object
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            
            # Should we show this bounding box?
            should_show_boundingbox = True
                        
            # We found a person
            if (object_name == "person"):
                personlastseentime = time.time() #cv2.getTickCount()
                personlastseentickcount = cv2.getTickCount()
                # if new - make this our primary person
                if (primary_person_found_in_last_image == 0):
                    print("New primary person found - choose costume")
                    greeting = random.choice(greetinglist)
                    costume  = random.choice(costumelist)
                    prompt   = random.choice(promptlist)
                    
                    ghostpositionx = xmax
                    ghostpositiony = ymin
                    
                    primary_person_center_x = int(xmin + (xmax - xmin)/2)
                    primary_person_center_y = int(ymin + (ymax - ymin)/2)
                    
                    chatterbox.speak(greeting + ", " + costume + ". " + prompt)

                    
                # Does this person overlap with our primary person?
                if ((primary_person_center_x > xmin and primary_person_center_x < xmax and primary_person_center_y > ymin and primary_person_center_y < ymax)):
                    
                    # Has the primary person been assigned in this picture yet?
                    if (primary_person_found == 0):
                        primary_person_found = 1
                        # get their center coordinates
                        primary_person_center_x = int(xmin + (xmax - xmin)/2)
                        primary_person_center_y = int(ymin + (ymax - ymin)/2)
                        color = (10, 255, 0)
						# Scale position from 0-1080 to 20-160
                        degree = (int) ((primary_person_center_x/1080)*140+20)

                        if (debug_tw ==1) :
                            print("x = " + str(primary_person_center_x) + ", y = " + str(primary_person_center_y))
                        
                        object_name = costume
                        
                        if debug_tw == 1:
                            cv2.circle(frame, (primary_person_center_x, primary_person_center_y), 5, color, -1)
                    else:
                        print("found person but will not show it") 
                        should_show_boundingbox = False
                        color = (255, 0, 0)
                        
                
            if(ghostpositionx > 0) :
                frame = picture_in_picture(frame, ghostimage, ghostpositionx, ghostpositiony)
            
            if (should_show_boundingbox or debug_tw == 1): 
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 6)
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fontsize, 2) # Get font size
                label_ymin = ymin + labelSize[1] + 10 #max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-2), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-2), cv2.FONT_HERSHEY_DUPLEX, fontsize, (0, 0, 0), 4) # Draw label text
                

    # Draw framerate in corner of frame
    if (debug_tw == 1):
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2,cv2.LINE_AA)

    if (primary_person_found == 1):
        out.write(frame)

    # All the results have been drawn on the frame, so it's time to display it.
    #if (xmin > 0):
	#    result_image = picture_in_picture(frame, image, xmin, ymin)
    #else:
	#    result_image = frame
		
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    personlosttickcount = cv2.getTickCount() - personlastseentickcount
    if (primary_person_found == 0 and primary_person_found_in_last_image == 1 and personlosttickcount > 5):
        primary_person_center_x = 0
        primary_person_center_y = 0

    personlosttime = time.time() - personlastseentime
    if (personlosttime > 1 and eyesopen == 1):
        print("Eyes closed")
        print("Person lost, time since seen: " + str(personlosttime) )
        out.release()
        eyesopen = 0
    elif (primary_person_found == 1 and eyesopen == 0):
        print("Eyes open")
        filename = "/home/orangepi/tflite1/Videos/video" + str(round(time.time())) + ".mp4"
        print("Filename: " + filename)
        out = cv2.VideoWriter(filename, fourcc, 20.0, (imW, imH))
        eyesopen = 1
        
    
    #print("Time since seen: " + str(personlosttime) )
    if (use_serial == 1) :
        if (eyesopen == 0):
            degree = -1
        #update eyes once per second
        if (time.time() - eyeslastupdated > 0.5):
            print("SERIAL: sending {}".format(degree))
            SerialObj.write(bytes('{}\n'.format(degree), 'utf-8'))
            eyeslastupdated = time.time()
    

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
out.release()
cv2.destroyAllWindows()
videostream.stop()
