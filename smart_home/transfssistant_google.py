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

import google.oauth2.credentials
import RPi.GPIO as GPIO
from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file
from voice_engine.source import Source
from voice_engine.doa_respeaker_4mic_array import DOA
from picamera import PiCamera

from src/networking import Connection

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)
GPIO.setup(6, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

cam_flag = True
camera = PiCamera()

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
        GPIO.output(13,True)
        if not doa is None: 
            direction = doa.get_direction()
            print('detected {} at direction {}'.format(keyword, direction))
        if cam_flag:
            
            camera.rotation = 180
            camera.capture("one-shot.jpg")
            networking.upload_images("192.168.1.69", 8888, ["one-shot.jpg"])
            cam_flag = False

    print(event)

    if ((event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']) or
            (event.type == EventType.ON_CONVERSATION_TURN_TIMEOUT) or
            (event.type == EventType.ON_NO_RESPONSE)):
        print()
        GPIO.output(13,False)

def main():
    doa = None
    # DoA
#    src = Source(rate=16000, frames_size=320, channels=4, device_index=0)
#    src = Source(rate=16000, frames_size=320, channels=4)
#    doa = DOA(rate=16000, chunks=20)
#    src.link(doa)
#    src.recursive_start()

    # GPIO
    GPIO.output(13,False)
    GPIO.output(26,True)
    GPIO.output(6,True)   

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
