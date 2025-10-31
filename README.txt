To get the program to run you need to install tensorflow lite with a simple object detection model. I used detect.tflite 

Files:
 Costumes.txt - a list of random costumes
 greeting.txt - a list of random greetings
 prompt.txt - a sentence ender or prompt

Greeting by text to speech (google) will be <Greeting> <Costume> <Prompt>. The bounding box of the person will have <Costume> as the label.

I have a howtostart.sh file that contains the following. Was hoping to be able to run this on startup - but was unsuccessful in getting that to work.


#!/usr/bin/env bash
#sleep 5
cd tflite1/
source tflite1-env/bin/activate
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
