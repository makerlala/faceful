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
import sys
import re
import os

from database import DataBase
from logger import Logger

# the settings has to be in data folder
PATH_TO_SETTINGS = "scripts/settings.conf"


class Settings:
    def __init__(self):
        self.path_to_settings_file = PATH_TO_SETTINGS
        self.path = ""
        self.project_path = ""
        self.fresh = True
        self.max_photo_fetch = 20
        self.face_cls_training_threshold = 2
        self.box_color = ""
        self.facedet_host = "127.0.0.1"
        self.facedet_port = 8888
        self.facedet_batch = 1
        self.facenet_src = ""
        self.facenet_model = ""
        self.facenet_classifier = ""
        self.facenet_classifier_images = ""
        self.facenet_classifier_image_size = 0
        self.facenet_classifier_batch_size = 0
        self.facedet_model = ""
        self.facedet_labels = ""
        self.tensorflow_models_src = ""
        self.tensorflow_object_detection_model = ""
        self.tensorflow_object_detection_labels = ""
        self.info = False
        self.debug = False
    
    def parse_settings(self, pairs):
        for pair in pairs:
            try:
                if pair[0] == "photospath":
                    self.path = pair[1]
                elif pair[0] == "projectpath":
                    self.set_project_path(pair[1])
                elif pair[0] == "fresh":
                    self.fresh = (pair[1] == "True")
                elif pair[0] == "maxphotofetch":
                    self.max_photo_fetch = int(pair[1])
                elif pair[0] == "faceclassifiertrainingthreshold":
                    self.face_cls_training_threshold = int(pair[1])
                elif pair[0] == "facedethost":
                    self.facedet_host = pair[1]
                elif pair[0] == "facedetport":
                    self.facedet_port = int(pair[1])
                elif pair[0] == "facedetbatch":
                    self.facedet_batch = int(pair[1])
                elif pair[0] == "facenetsrc":
                    self.facenet_src = self.append_to_project_path(pair[1])
                elif pair[0] == "facenetmodel":
                    self.facenet_model = self.append_to_project_path(pair[1])
                elif pair[0] == "facenetclassifier":
                    self.facenet_classifier = self.append_to_project_path(pair[1])
                elif pair[0] == "facenetclassifierimages":
                    self.facenet_classifier_images = self.append_to_project_path(pair[1])
                elif pair[0] == "facenetclassifiertrainimagesize":
                    self.facenet_classifier_image_size = int(pair[1])
                elif pair[0] == "facenetclassifiertrainbatchsize":
                    self.facenet_classifier_batch_size = int(pair[1])
                elif pair[0] == "tensorflowmodelssrc":
                    self.tensorflow_models_src = self.append_to_project_path(pair[1])
                elif pair[0] == "tensorflowobjectdetectionmodel":
                    self.tensorflow_object_detection_model = self.append_to_project_path(pair[1])
                elif pair[0] == "tensorflowobjectdetectionlabels":
                    self.tensorflow_object_detection_labels = self.append_to_project_path(pair[1])
                elif pair[0] == "facedetmodel":
                    self.facedet_model = self.append_to_project_path(pair[1])
                elif pair[0] == "facedetlabels":
                    self.facedet_labels = self.append_to_project_path(pair[1])
                elif pair[0] == "info":
                    self.info = (pair[1] == "True")
                elif pair[0] == "debug":
                    self.debug = (pair[1] == "True")
                else:
                    Logger.warning("Unknown settings: " + str(pair[0]))
            except Exception as e:
                Logger.error("Exception in parse settings for key " + pair[0] + ": " + str(e))
                sys.exit()
        Logger.debug("Settings look ok!")
                
    def load_from_db(self):
        db = DataBase()
        pairs = db.get_settings()
        self.parse_settings(pairs)
        
    def load_from_file(self):
        Logger.info("Loading settings from " + PATH_TO_SETTINGS)
        pairs = []
        with open(PATH_TO_SETTINGS,'rt') as f:
            for line in f.readlines():
                tokens = re.sub(r"\s","", line).split("=")
                if len(tokens) != 2:
                    Logger.error("Error in settings file '" + PATH_TO_SETTINGS + "' at line '" + line + "'")
                    break
                pairs.append(tokens)
            f.close()
        self.parse_settings(pairs)

    def set_project_path(self, path):
        if not path.endswith("/"):
            self.project_path = path + "/"
        else:
            self.project_path = path
        if not os.path.isdir(self.project_path):
            Logger.fatal("Project path not found: " + self.project_path)

    def append_to_project_path(self, relative_path):
        path = self.project_path + relative_path
        if not os.path.isdir(self.project_path):
            Logger.fatal("Not a folder: " + path)
        return path

    @staticmethod
    def load():
        set_file = Settings()
        set_file.load_from_file()
        return set_file
