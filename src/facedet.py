'''
Copyright (c) 2018 - Dumi Loghin (dumi@makerlala.com)

This file is part of faceful - an open source smart photo gallery with 
object and face recognition. 

Parts of this code interacts and has been inspired by facenet project 
by David Sandberg which is licensed under MIT licence 
(see https://github.com/davidsandberg/facenet).

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import cv2
from scipy import misc
import csv
import socket
import signal
from pprint import pprint

from networking import Connection
from settings import Settings
from webbrowser import Opera

global settings
settings = Settings.load()

print("facenet source folder: " + settings.facenet_src)
sys.path.append(settings.facenet_src)
import align.detect_face
import facenet

print("tensorflow models source folder: " + settings.tensorflow_models_src)
sys.path.append(settings.tensorflow_models_src)
from object_detection.utils import label_map_util

# Networking
HOST = ''
PORT = settings.facedet_port

# Face Detection
PATH_TO_FACEDET_MODEL = settings.facedet_model
PATH_TO_FACEDET_LABELS = settings.facedet_labels
NUM_CLASSES = 2

# Face classification
PATH_TO_FACENET_MODEL = settings.facenet_model
PATH_TO_FACENET_CLASSIFIER = settings.facenet_classifier
PATH_TO_LFW_IMAGES = settings.facenet_classifier_images
TRAIN_BATCH_SIZE = settings.facenet_classifier_batch_size
TRAIN_IMG_SIZE = settings.facenet_classifier_image_size

# Object classification
PATH_TO_OBJDET_MODEL = settings.tensorflow_object_detection_model
PATH_TO_OBJDET_LABELS = settings.tensorflow_object_detection_labels
NUM_CLASSES_OBJDET = 90

FACEDET_THRESH = 0.65

sock = None
server_flag = True

def sigint_handler(signal, frame):
    global sock
    global server_flag
    print('You pressed Ctrl+C. Server will stop.')
    sock.shutdown(socket.SHUT_WR)
    sock.close()
    server_flag = False

signal.signal(signal.SIGINT, sigint_handler)

''' Get a square bounding box by extending the existing box 
into a single direction. '''
def get_square_box_v1(x0, y0, x1, y1, max_w, max_h):
    wb = x1 - x0
    hb = y1 - y0
    if wb < hb:
        if x1 + (hb - wb) >= max_w:
            if x0 - (hb - wb) < 0:
                x0 = 0
            else:
                x0 = x0 - (hb - wb)
        else:
            x1 = x1 + (hb - wb)
    elif wb > hb:
        if y1 + (wb - hb) >= max_h:
            if y0 - (wb - hb) < 0:
                y0 = 0
            else:
                y0 = y0 - (wb - hb)
        else:
            y1 = y1 + (wb - hb)

    return [x0, y0, x1, y1]

''' Get a square bounding box by extending the existing box 
both to the left and right, or both to the top and bottom, 
as needed. '''
def get_square_box(x0, y0, x1, y1, max_w, max_h):
    wb = x1 - x0
    hb = y1 - y0
    if wb < hb:
        d0 = (hb - wb) // 2
        if 2 * d0 < hb - wb:
            d1 = d0 + 1
        else:
            d1 = d0
        if x1 + d1 >= max_w:
            d1 = max_w - x1 - 1
            d0 = hb - wb - d1
            if x0 - d0 < 0:
                x0 = 0
                x1 = max_w - 1
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                x0 = x0 - d0
                x1 = x1 + d1
        else:
            x1 = x1 + d1
            if x0 - d0 < 0:
                x0 = 0
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                x0 = x0 - d0

    elif wb > hb:
        d0 = (wb - hb) // 2
        if 2 * d0 < wb - hb:
            d1 = d0 + 1
        if y1 + d1 >= max_h:
            d1 = max_h - y1 - 1
            d0 = wb - hb - d1
            if y0 - d0 < 0:
                y0 = 0
                y1 = max_h - 1
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                y0 = y0 - d0
                y1 = y1 + d1
        else:
            y1 = y1 + d1
            if y0 - d0 < 0:
                y0 = 0
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                y0 = y0 - d0

    return [x0, y0, x1, y1]

'''Load an image into memory'''
def load_image(image_path):
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:        
        print('{}: {}'.format(image_path, e))
        return None

    if img.ndim < 2:
        print('Unable to align "%s"' % image_path)
        return None
    elif img.ndim == 2:
        img = facenet.to_rgb(img)
    elif len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return img

def get_roi(image, bounding_boxes):
    border = 20
    nrof_faces = bounding_boxes.shape[0]
    [h, w] = np.asarray(image.shape)[0:2]
    faces = []
    boxes = []
    if nrof_faces < 1:
        return [], []
                
    for i in range(nrof_faces):
        det = np.squeeze(bounding_boxes[i,0:4])
        x0 = max(int(det[0]) - border, 0)
        x1 = min(int(det[2]) + border, w-1)
        y0 = max(int(det[1]) - border, 0)
        y1 = min(int(det[3]) + border, h-1)
                    
        [x0, y0, x1, y1] = get_square_box(x0, y0, x1, y1, w, h)
        print(str(x0) + " " + str(y0) + " " + str(x1) + " " + str(y1))                    
        cropped = image[y0:y1,x0:x1,:]
        scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
        prew = facenet.prewhiten(scaled)
        faces.append(prew)
        boxes.append([x0, y0, x1, y1])
        # misc.imsave("roi" + str(i) + ".png", prew)

    return faces, boxes

def facedet_as_service():
    print("Running as a service...")
    # networking
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, PORT))
    except socket.error as msg:
        print("Socket bind failed. Error code : " + str(msg[0]) + ", message " + msg[1])
        return
    sock.listen(10)
    
    # load models
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    g_detection = tf.Graph()
    g_facenet = tf.Graph()

    with g_detection.as_default():
        print('Loading face detection model')
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FACEDET_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')        
        image_tensor = g_detection.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes_tensor = g_detection.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores_tensor = g_detection.get_tensor_by_name('detection_scores:0')
        classes_tensor = g_detection.get_tensor_by_name('detection_classes:0')
        num_detections_tensor = g_detection.get_tensor_by_name('num_detections:0')

    with g_facenet.as_default():
        print('Loading feature extraction model')
        facenet.load_model(PATH_TO_FACENET_MODEL)
        images_placeholder = g_facenet.get_tensor_by_name("input:0")
        embeddings = g_facenet.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = g_facenet.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        print('Loading face classifier: ' + PATH_TO_FACENET_CLASSIFIER)
        with open(PATH_TO_FACENET_CLASSIFIER, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
# 1. Detect Face
#    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
#    minsize = 20 # minimum size of face
#    threshold = [ 0.8, 0.85, 0.85 ]  # three steps's threshold
#    factor = 0.709 # scale factor

    while server_flag:
        print('Waiting for a connection...')
        try:
            conn, addr = sock.accept()
        except socket.error as msg:
            break
        startTime = time.time()
        img_data, img_paths = Connection.download_images(conn, in_mem = True)
        print('Image download took {} s'.format(time.time() - startTime))
        print('Received {} images'.format(len(img_data)))

        with open('trfcam.jpg', 'wb') as trfile:
            trfile.write(img_data[0])
            trfile.close()

        faces = []
        boxes = []
        startTime = time.time()
        with tf.Session(config = config, graph = g_detection) as sess:
            np.random.seed(777)

#            for img_path in img_paths:
                # load image
                # img = load_image(img_path)
            for img_buf in img_data:                
                img = cv2.imdecode(np.asarray(bytearray(img_buf), dtype=np.uint8), 0)
                if img.ndim < 2:
                    print('Unable to align image')
                    continue
                elif img.ndim == 2:
                    img = facenet.to_rgb(img)
                elif len(img.shape) > 2 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)               

                [h, w] = np.asarray(img.shape)[0:2]
                # bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                image_np_expanded = np.expand_dims(img, axis=0)
                (bounding_boxes, scores, classes, num_detections) = sess.run([boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor], feed_dict={image_tensor: image_np_expanded})
                bounding_boxes = np.squeeze(bounding_boxes)  
                scores = np.squeeze(scores)                
                nrof_faces = bounding_boxes.shape[0]
                n = 0
                for i in range(nrof_faces):
                    # Using align
                    # det = np.squeeze(bounding_boxes[i,0:4])

                    # Using face detection
                    if scores[i] < 0.75:
                        continue                    
                    det = [0, 0, 0, 0]
                    det[0] = bounding_boxes[i,1] * w
                    det[1] = bounding_boxes[i,0] * h
                    det[2] = bounding_boxes[i,3] * w
                    det[3] = bounding_boxes[i,2] * h                   

                    x0 = max(int(det[0]) - 20, 0)
                    x1 = min(int(det[2]) + 20, w-1)
                    y0 = max(int(det[1]) - 20, 0)
                    y1 = min(int(det[3]) + 20, h-1)
                    
                    [x0, y0, x1, y1] = get_square_box(x0, y0, x1, y1, w, h)                                     
                    cropped = img[y0:y1,x0:x1,:]
                    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                    prew = facenet.prewhiten(scaled)
                    faces.append(prew)
                    boxes.append([x0, y0, x1, y1])
                    misc.imsave("roi" + str(n) + ".png", prew)
                    n = n + 1
                
        print('Face detection took {} s'.format(time.time() - startTime))
        nrof_faces = len(faces)
        print('Detected {} faces'.format(nrof_faces))
        if nrof_faces == 0:
            continue

        startTime = time.time()
        with tf.Session(config = config, graph = g_facenet) as sess:
            np.random.seed(666)

            emb_array = np.zeros((len(faces), embedding_size))
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            emb_array[:,:] = sess.run(embeddings, feed_dict=feed_dict)
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
            print('Face classification took {} s'.format(time.time() - startTime))

            with open('facedet_front.txt','wt') as outfile:
                for i in range(len(best_class_indices)):
                    # print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    outfile.write(class_names[best_class_indices[i]] + "\n")
                outfile.close()

    sock.close()
    print("Service stopped.")

def facedet_objdet_as_service():
    print("Running face and object detection as a service...")
    
    # networking
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, PORT))
    except socket.error as msg:
        print("Socket bind failed. Error code : " + str(msg[0]) + ", message " + msg[1])
        return
    sock.listen(10)
    
    # load models
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    g_face_det = tf.Graph()
    g_obj_det = tf.Graph()
    g_facenet = tf.Graph()

    with g_face_det.as_default():
        print('Loading face detection model: ' + PATH_TO_FACEDET_MODEL)
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FACEDET_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')        
        image_tensor_face = g_face_det.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes_tensor_face = g_face_det.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores_tensor_face = g_face_det.get_tensor_by_name('detection_scores:0')
        classes_tensor_face = g_face_det.get_tensor_by_name('detection_classes:0')
        num_detections_tensor_face = g_face_det.get_tensor_by_name('num_detections:0')

    with g_obj_det.as_default():
        print('Loading object detection model: ' + PATH_TO_OBJDET_MODEL)
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_OBJDET_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # Definite input and output Tensors for detection_graph
        image_tensor_obj = g_obj_det.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes_obj = g_obj_det.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores_obj = g_obj_det.get_tensor_by_name('detection_scores:0')
        detection_classes_obj = g_obj_det.get_tensor_by_name('detection_classes:0')
        num_detections_obj = g_obj_det.get_tensor_by_name('num_detections:0')
    
    with g_facenet.as_default():
        print('Loading feature extraction model: ' + PATH_TO_FACENET_MODEL)
        facenet.load_model(PATH_TO_FACENET_MODEL)
        faces_placeholder = g_facenet.get_tensor_by_name("input:0")
        embeddings = g_facenet.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = g_facenet.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        print('Loading face classifier: ' + PATH_TO_FACENET_CLASSIFIER)
        with open(PATH_TO_FACENET_CLASSIFIER, 'rb') as infile:
            (facenet_model, facenet_class_names) = pickle.load(infile)
    
    label_map = label_map_util.load_labelmap(PATH_TO_OBJDET_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES_OBJDET, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    ''' Server '''
    while server_flag:
        print('Waiting for connections...')
        try:
            conn, addr = sock.accept()
        except socket.error as msg:
            break
        
        startTime = time.time()
        try:
            operation = conn.recv(32).decode()
            if operation == "train":
                conn.send("OK".encode())
                try:
                    tokens = conn.recv(32).decode().split(' ')
                    no_objects = int(tokens[0])
                    no_images = int(tokens[1])
                except Exception as e:
                    conn.send("Error with integer values: " + str(e))
                    continue
                conn.send("OK".encode())
                print("Request to perform training on {} aliases with {} images each.".format(no_objects, no_images))
                for i in range(no_objects):
                    alias = conn.recv(32).decode().replace(" ","_")
                    print("Alias: " + alias)
                    folder = PATH_TO_LFW_IMAGES + "/" + alias
                    if os.path.exists(folder):
                        os.rmdir(folder)
                    os.makedirs(folder)
                    conn.send("OK".encode())
                    img_data, _ = Connection.download_images(conn, in_mem = True, close_conn = False)
                    n = 0
                    for img_buf in img_data:
                        img = cv2.imdecode(np.asarray(bytearray(img_buf), dtype=np.uint8), 0)
                        cv2.imwrite(folder + "/" + alias + str(n) + ".png", img)
                        
                dataset = facenet.get_dataset(PATH_TO_LFW_IMAGES)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / TRAIN_BATCH_SIZE))
                emb_array = np.zeros((nrof_images, embedding_size))
                with tf.Session(config = config, graph = g_facenet) as sess:
                    for i in range(nrof_batches_per_epoch):
                        start_index = i * TRAIN_BATCH_SIZE
                        end_index = min((i+1) * TRAIN_BATCH_SIZE, nrof_images)
                        paths_batch = paths[start_index:end_index]
                        images = facenet.load_data(paths_batch, False, False, TRAIN_IMG_SIZE)
                        feed_dict = { faces_placeholder:images, phase_train_placeholder:False }
                        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                with open(PATH_TO_FACENET_CLASSIFIER, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % PATH_TO_FACENET_CLASSIFIER)
                
            elif operation == "objdet" or operation == "facerec":
                conn.send("OK".code())
                img_data, _ = Connection.download_images(conn, in_mem = True, close_conn = False)
            else:
                conn.send("No such operation".encode())
        except Exception as e:
            print(str(e))
        print('Images download took {} s'.format(time.time() - startTime))
        print('Received {} images'.format(len(img_data)))        

        send_buf = ""
        startTime = time.time()
        images = []
        for img_buf in img_data:
            if img_buf == None or len(img_buf) == 0:
                continue 
            img = cv2.imdecode(np.asarray(bytearray(img_buf), dtype=np.uint8), 0)
            if img.ndim < 2:
                print('Unable to align image')
                continue
            elif img.ndim == 2:
                img = facenet.to_rgb(img)
            elif len(img.shape) > 2 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)               

            [h, w] = np.asarray(img.shape)[0:2]
            
            if operation == "objdet":
                image_np_expanded = np.expand_dims(img, axis=0)
                images.append(image_np_expanded)
            elif operation == "facerec":  
                scaled = misc.imresize(img, (160, 160), interp='bilinear')
                prew = facenet.prewhiten(scaled)
                images.append(prew)
            
        if operation == "objdet":
            for image_np_expanded in images:
                # Face detection - running in both cases
                with tf.Session(config = config, graph = g_face_det) as sess:
                    np.random.seed(777)
                    (bounding_boxes, scores, classes, num_detections) = sess.run([boxes_tensor_face, scores_tensor_face, classes_tensor_face, num_detections_tensor_face], feed_dict={image_tensor_face: image_np_expanded})
                    bounding_boxes = np.squeeze(bounding_boxes)  
                    scores = np.squeeze(scores)                 
                    for i in range(bounding_boxes.shape[0]):                    
                        if scores[i] < FACEDET_THRESH:
                            continue
                    send_buf = send_buf + "face {} {} {} {};".format(bounding_boxes[i,1], bounding_boxes[i,0], bounding_boxes[i,3], bounding_boxes[i,2])
                    
                # Object detection
                with tf.Session(config = config, graph = g_obj_det) as sess:
                    np.random.seed(777)
                    (bounding_boxes, scores, classes, num_detections) = sess.run([detection_boxes_obj, detection_scores_obj, detection_classes_obj, num_detections_obj], feed_dict={image_tensor_obj: image_np_expanded})
                    bounding_boxes = np.squeeze(bounding_boxes)  
                    scores = np.squeeze(scores)
                    classes = np.squeeze(classes)              
                    for i in range(bounding_boxes.shape[0]):                    
                        if scores[i] < FACEDET_THRESH:
                            continue
                    send_buf = send_buf + "{} {} {} {} {};".format(category_index[int(classes[i])]['name'], bounding_boxes[i,1], bounding_boxes[i,0], bounding_boxes[i,3], bounding_boxes[i,2])                
            
        elif operation == "facerec":
            with tf.Session(config = config, graph = g_facenet) as sess:
                feed_dict = { faces_placeholder:images, phase_train_placeholder:False }
                emb_array = np.zeros((len(img_data), embedding_size))
                emb_array[:] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = facenet_model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                for i in range(len(best_class_indices)):
                    send_buf = send_buf +  "%d  %s %.3f;" % (i, facenet_class_names[best_class_indices[i]], best_class_probabilities[i])
                    
        print('Operation {} took {} s'.format(operation, time.time() - startTime))
        print('Sending: ' + send_buf)
        conn.send(str.encode(send_buf))
        conn.send(str.encode("END"))

    sock.close()
    print("Service stopped.")

'''Detect faces on all images from all cameras'''
def facedet_bulk(args):
    if len(args) < 3:
        print("Usage: " + args[0] + " <settings_file> <camera_config_file>")
        return
    
    start_time = time.time()

    img_paths = []
    img_loc = []
    img_tstamp = []
    with open(args[2], 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[0].startswith('#'):
                continue
            img_paths.append(row[0] + ".jpg")
            img_loc.append(row[6] + ' ' + row[7])
            with open(row[0] + ".txt", 'rt') as tsfile:
                img_tstamp.append(int(tsfile.read()))
                tsfile.close()
        csvfile.close()
    
    faces_file = ""
    with open(args[1], 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[0].startswith('#'):
                continue
            if row[1] == "facedet file":
                faces_file = row[1]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():      
        with tf.Session(config = config) as sess:
            
            np.random.seed(666)

            # 0. Read images
            img_data = []
            min_dim = 1000000000
            for img_path in img_paths:
                img = load_image(img_path)
                img_data.append(img)
                [h, w] = np.asarray(img.shape)[0:2]
                min_dim = min(min_dim, h, w)

            # 1. Detect Faces
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
            min_size = 20 # minimum size of face
            threshold = [ 0.8, 0.85, 0.85 ]  # three steps's threshold
            factor = 0.709 # scale factor
            
            bounding_boxes = align.detect_face.bulk_detect_face(img_data, min_size/min_dim, pnet, rnet, onet, threshold, factor)
            
            print("Total size of face detection list: " + str(len(bounding_boxes)))

            faces = []
            boxes = []
            img_index = []
            n = -1
            
            for bounding_box in bounding_boxes:
                n = n + 1

                if bounding_box == None:
                    continue

                this_faces, this_boxes = get_roi(img_data[n], bounding_box[0])                
                for i in range(len(this_faces)):
                    faces.append(this_faces[i])
                    boxes.append(this_boxes[i])
                    img_index.append(n)
            
            if len(faces) == 0:
                with open(faces_file,'wt') as outfile:                    
                    outfile.write("I can see nobody")
                    outfile.close()
                return

            # 2. Recognize Face
  
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(PATH_TO_FACENET_MODEL)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((len(faces), embedding_size))
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            emb_array[:,:] = sess.run(embeddings, feed_dict=feed_dict)
            with open(PATH_TO_FACENET_CLASSIFIER, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                #predictions = model.predict(emb_array)
                #best_class_indices = predictions
                #best_class_probabilities = predictions
                with open('faces_file.txt','wt') as outfile:                    
                    outfile.write("I can see ")
                    for i in range(len(best_class_indices)):
                    # print('%4d  %s %s: %.3f' % (i, class_names[best_class_indices[i]], img_loc[img_index[i]], best_class_probabilities[i]))                    
                        outfile.write(class_names[best_class_indices[i]] + " at " + img_loc[img_index[i]] + " and ")
                    outfile.write(" that's it")
                    outfile.close()

    print('Face detection took {} s'.format(time.time() - start_time))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "-s":
            facedet_as_service()
        elif sys.argv[1] == "-o":
            facedet_objdet_as_service()
    else:
        facedet_bulk(sys.argv)
