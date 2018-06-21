'''
Copyright (c) 2018 - Dumi Loghin (dumi@makerlala.com)

This file is part of faceful - an open source smart photo gallery with 
object and face recognition.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
from __future__ import print_function

import argparse
import os
import os.path
import json
import time

import google.oauth2.credentials
import RPi.GPIO as GPIO
from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file
from voice_engine.source import Source
from voice_engine.doa_respeaker_4mic_array import DOA
from picamera import PiCamera
import numpy as np
import cv2

from pixels import pixels
# from src/networking import Connection

led1_port = 26
led2_port = 6
led_gnd_port = 13
motion_port = 16
pwm_port = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(led1_port, GPIO.OUT)
GPIO.setup(led2_port, GPIO.OUT)
GPIO.setup(led_gnd_port, GPIO.OUT)

cam_flag = True
camera = PiCamera()
camera.rotation = 180

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

def process_event(event, doa):
    """Pretty prints events.
    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.
    Args:
        event(event.Event): The current event to process.
    """
    global cam_flag

    if event.type == EventType.ON_NO_RESPONSE:
        cam_flag = True

    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()
        GPIO.output(led_gnd_port,True)
        if not doa is None: 
            direction = doa.get_direction()
            print('detected voice at direction {}'.format(direction))
            pixels.wakeup(direction)
        if cam_flag:
            tstamp = str(int(time.time()))
            imgfile = "one-shot-" + tstamp + ".jpg"
            detfile = "det-shot-" + tstamp + ".jpg"
            camera.capture(imgfile)

            img = cv2.imread(imgfile)            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,      
                minSize=(20, 20)
            )

            print('detected ' + str(len(faces)) + ' faces')
 
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                eyes = eyeCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor= 1.5,
                    minNeighbors=5,
                    minSize=(5, 5),
                )
        
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor= 1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )
        
                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

            cv2.imwrite(detfile, img)

#            networking.upload_images("192.168.1.69", 8888, ["one-shot.jpg"])
#            cam_flag = False

    print(event)

    if ((event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']) or
            (event.type == EventType.ON_CONVERSATION_TURN_TIMEOUT) or
            (event.type == EventType.ON_NO_RESPONSE)):
        print()
        GPIO.output(led_gnd_port,False)
        pixels.off()

def main():
    doa = None
    # DoA
#    src = Source(rate=16000, frames_size=320, channels=4, device_index=0)
    src = Source(rate=16000, frames_size=320, channels=4)
    doa = DOA(rate=16000, chunks=20)
    src.link(doa)
    src.recursive_start()

    # GPIO
    GPIO.output(led_gnd_port,False)
    GPIO.output(led1_port,True)
    GPIO.output(led2_port,True)   

    # Google Assistant
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--credentials', type=existing_file,
                        metavar='OAUTH2_CREDENTIALS_FILE',
                        default=os.path.join(
                            os.path.expanduser('/home/pi/.config'),
                            'google-oauthlib-tool',
                            'credentials.json'
                        ),
                        help='Path to store and read OAuth2 credentials')
    args = parser.parse_args()
    with open(args.credentials, 'r') as f:
        credentials = google.oauth2.credentials.Credentials(token=None, **json.load(f))

    with Assistant(credentials,"RPi3_Assistant") as assistant:
        for event in assistant.start():
            process_event(event, doa)

#    src.recursive_stop()

if __name__ == '__main__':
    main()
