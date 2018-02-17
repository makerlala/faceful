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
import os.path
import os
import sys
import re
import cv2
import numpy as np
from flask import Flask, session, request, redirect
from flask import send_file
from flask.templating import render_template
from multiprocessing import Pool

proc_pool = None
training_in_progress = False

from networking import Connection
from settings import Settings
from database import DataBase

settings = None

'''
Entity: folder, image, video
- arranged as tree for folder navigation
- arranged as map for quick access
- has an id in this app (en_id)
- has a distinct id in the database (db_id)
(that is why we keep two maps, entity_id_map and entity_db_id_map
'''
root_entity = None
entity_id_map = {}
entity_db_id_map ={}
root_en_id = 0

class Entity:
    def __init__(self, path, name, en_id, db_id, is_dir, is_video):
        self.path = path
        self.name = name
        self.en_id = en_id
        self.prev_id = -1
        self.next_id = -1
        self.db_id = db_id
        self.is_dir = is_dir
        self.is_video = is_video
        self.children = []
    
    def is_dir(self):
        return self.is_dir
    
    def is_video(self):
        return self.is_video
    
    def add_child(self, child):
        self.children.append(child)
    
    def find_entity_in_tree_with_id(self, en_id):
        if self.en_id == en_id:
            return self
        for child in self.children:
            found = child.find_entity_in_tree_with_id(en_id)
            if found is not None:
                return found
        return None
    
    def find_entity_in_tree_with_db_id(self, db_id):
        if self.db_id == db_id:
            return self
        for child in self.children:
            found = child.find_entity_in_tree_with_db_id(db_id)
            if found is not None:
                return found
        return None
    
    '''
    Build entities tree recursively
    path - real path in the filesystem
    vpath - virtual path in the static folder
    db - DataBase object
    name - node name (file name)
    '''
    @staticmethod
    def build_tree(path, vpath, db, name):
        global settings
        global root_en_id
        global entity_id_map
        global entity_db_id_map
        root = Entity(vpath, name, root_en_id, -1, True, False)
        entity_id_map[root_en_id] = root
        root_en_id = root_en_id + 1
        prev_child = None
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            virtual_file_path = vpath + "/" + file
            if os.path.isdir(file_path):
                child = Entity.build_tree(file_path, virtual_file_path, db, file)
                root.add_child(child)
            elif file.startswith("."):
                continue
            elif is_video(file) or is_photo(file):
                db_id = db.get_photo_key(file_path)
                child = Entity(virtual_file_path, file, root_en_id, db_id, False, is_video(file))
                if prev_child is not None:
                    child.prev_id = prev_child.en_id
                    prev_child.next_id = child.en_id
                prev_child = child
                root.add_child(child)
                entity_id_map[root_en_id] = child
                root_en_id = root_en_id + 1
                entity_db_id_map[db_id] = child
            else:
                if settings.info:
                    print("Unknown file: " + file_path)
        return root


        
def is_photo(f):
    f = str.lower(f)
    return f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".bmp") or f.endswith(".gif")
    
def is_video(f):
    f = str.lower(f)
    return f.endswith(".mpg") or f.endswith(".avi") or f.endswith(".mp4") or f.endswith(".mov") or f.endswith(".wmv")

def update_database():
    global settings
    
    db = DataBase()
    db.create()
    
    img_paths = []
    
    conn = Connection(settings.facedet_host, settings.facedet_port)
    
    for (dirpath, dirnames, filenames) in os.walk(settings.path):
        for f in filenames:
            if is_photo(f):
                img_paths.append(os.path.join(dirpath, f))
    
    n = len(img_paths)
    i = 1.0
    for img_path in img_paths:
        print("Progress {}".format(100.8*i/n))
        i = i + 1.0      
        img_id = db.get_photo_key(img_path)
        if img_id != -1:
            print("Photo is already in the database")
            continue

        img_id = db.insert_photo(img_path)
        print(img_path)
        ret = conn.upload_images(settings.facedet_host, settings.facedet_port, [img_path])        
        records = ret.split(';')
        for record in records:
            if record == "END":
                break
            tokens = record.split(' ')
            db.insert_box(img_id, tokens[1], tokens[2], tokens[3], tokens[4], tokens[0])
            
    db.close()
    conn.close()

def train_classifier_callback(result):
    global training_in_progress
    training_in_progress = False
    print("Training classifier is done.")
    
def train_classifier():
    global settings
    print("Start training classifier")
    conn = Connection(settings.facedet_host, settings.facedet_port)
    db = DataBase()
    marked_faces = db.get_boxes_with_labels(labelIdx = 7)
    faces = {}
    for alias in marked_faces.keys():
        faces[alias] = []
        for i in range(settings.face_cls_training_threshold):
            box = marked_faces[alias][i]
            photo = db.get_photo(box[1])
            img = cv2.imread(photo[1])
            [h, w] = np.asarray(img.shape)[0:2]
            x0 = int(float(box[2]) * float(w))
            y0 = int(float(box[3]) * float(h))
            x1 = int(float(box[4]) * float(w))
            y1 = int(float(box[5]) * float(h))
            img = cv2.resize(img[y0:y1,x0:x1,:], (settings.facenet_classifier_image_size, settings.facenet_classifier_image_size))
            try:
                faces[alias].append(cv2.imencode(".png", img)[1].tostring())
            except Exception as e:
                print("Exception in train_classifier" + str(e))    
    
    resp = conn.send_message("train")
    print(resp)
    if resp != "OK":
        return
    
    data = "{} {}".format(len(faces.keys()), settings.face_cls_training_threshold)
    print("Send: " + data)
    resp = conn.send_message(data)
    print(resp)
    if resp != "OK":
        return
    
    for alias in faces.keys():
        resp = conn.send_message(alias)
        print(resp)
        if resp != "OK":
            return
        conn.upload_images(faces[alias], in_mem = True)
            
    conn.close()
        
def update_faces(marked_faces):
    db = DataBase()
    for box in db.get_boxes_with_faces():
        photo = db.get_photo(box[1])
        img = cv2.imread(photo[1])
        [h, w] = np.asarray(img.shape)[0:2]
        x0 = int(float(box[2]) * float(w))
        y0 = int(float(box[3]) * float(h))
        x1 = int(float(box[4]) * float(w))
        y1 = int(float(box[5]) * float(h))
        roi = img[y0:y1,x0:x1,:]
        # TODO - do

def scale_image(image_path, image_id, scaled_h, scaled_w, new_image_path, minRatio=False): 
    img = cv2.imread(image_path)
    [h, w] = np.asarray(img.shape)[0:2]
    if minRatio:
        ratio = min(scaled_h/h, scaled_w/w)
    else:
        ratio = float(scaled_w)/float(w)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    cv2.imwrite(new_image_path, img)
    return new_image_path

def scale_boxes_image(image_path, image_id, scaled_h, scaled_w, new_image_path, boxes): 
    img = cv2.imread(image_path)
    [h, w] = np.asarray(img.shape)[0:2]
    ratio = min(scaled_h/h, scaled_w/w)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    [h, w] = np.asarray(img.shape)[0:2]
    n = 1
    for box in boxes:
        try:
            x0 = int(float(box[2]) * float(w))
            y0 = int(float(box[3]) * float(h))
            x1 = int(float(box[4]) * float(w))
            y1 = int(float(box[5]) * float(h))
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,0,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if box[6] == "face":
                txt = box[7] + " (box " + str(n) + ")"
            else:
                txt = box[6] + " (box " + str(n) + ")"
            n= n + 1
            cv2.putText(img, txt, (x0,y0-10), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        except Exception as e:
            print("Error in scale_boxes_image: " + str(e))
    cv2.imwrite(new_image_path, img)
    return new_image_path

def get_filename_cached_photo(photo_id):
    return "static/tmp/photo_" + str(photo_id) + ".png"

def get_filename_cached_photo_labels(photo_id):
    return "static/tmp/photo_labels_" + str(photo_id) + ".png"
    
def get_thumbnail(image_path, image_id, scaled_w):
    filename = "static/tmp/thumbnail_" + str(image_id) + ".png"
    if os.path.isfile(filename):
        return filename
    return scale_image(image_path, image_id, 0, scaled_w, filename)

def get_photo(image_path, image_id, window_h, window_w, boxes, withLabels=False):
    if withLabels:
        filename = get_filename_cached_photo_labels(image_id)
    else:
        filename = get_filename_cached_photo(image_id)
    if os.path.isfile(filename):
        return filename
    if withLabels:
        return scale_boxes_image(image_path, image_id, 0.9 * window_h, 0.6 * window_w, filename, boxes)
    else:
        return scale_image(image_path, image_id, 0.9 * window_h, 0.6 * window_w, filename, minRatio=True)

''' ============================= '''
''' === === === Flask === === === '''
    
app = Flask("My Smart Gallery")
app.secret_key = 'uaYX3T283RGDW384T1EGVas1363'

@app.route('/', methods=['GET'])
def index():
    global settings
    global root_entity
    global entity_id_map
    path_id = request.args.get('path_id')
    if path_id is not None:
        path_id = int(path_id)
        entity = entity_id_map[path_id]
        print(entity)
        if entity is not None:
            session['lastIndex' + str(path_id)] = settings.max_photo_fetch
            return render_template('index.html', 
                                   entities = entity.children, 
                                   path_id = path_id, 
                                   maxfetch = settings.max_photo_fetch)
            
    return render_template('index.html', 
                           entities = root_entity.children, 
                           path_id = -1, 
                           maxfetch = settings.max_photo_fetch)

@app.route('/getmorephotos', methods=['POST'])
def getmorephotos():
    global settings
    global root_entity
    global entity_id_map
    path_id = request.form.get('path_id')
    if path_id is not None:
        path_id = int(path_id)
        entity = entity_id_map[path_id]
        if entity is None:
            return '';
        lastIdxKey = 'lastIndex' + str(path_id)
        if lastIdxKey in session:
            lastIdx = session[lastIdxKey]
        else:
            lastIdx = 0
        data = ""
        nextIdx = min(lastIdx + settings.max_photo_fetch, len(entity.children))
        session[lastIdxKey] = nextIdx
        for i in range(lastIdx, nextIdx):
            child = entity.children[i]
            if child.is_video:
                data = data + '<div class="image fit"><a href="/detail?path_id=' + str(child.en_id) + '"><img src="static/img/icon_video_red_512px.png" alt="' + child.name + '" /></a</div>'
            else:
                data = data + '<div class="image fit"><a href="/detail?path_id=' + str(child.en_id) + '"><img src="/thumbnail?path_id=' + str(child.en_id) + '&w=400" alt="' + child.name + '" /></a></div>'
        return data
            
@app.route('/detail', methods=['GET'])
def detail():
    path_id = request.args.get('path_id')
    if path_id is None:
        return "Not found!"
    
    path_id = int(path_id)
    entity = entity_id_map[path_id]
    if entity is None:
        return "Not found!"
    
    if 'windowHeight' in session:
        wh = float(session['windowHeight'])
    else:
        wh = 600
    if 'windowWidth' in session:
        ww = float(session['windowWidth'])
    else:
        ww = 800
    
    if entity.is_video:
        return render_template('detail.html', 
                               entity = entity, 
                               width = ww, 
                               imgpath = "", 
                               stories = [], 
                               boxes = [])
    
    db = DataBase()
    boxes = db.get_boxes(entity.db_id)
    print(boxes)
    photopath = get_photo(entity.path, entity.en_id, wh, ww, boxes, withLabels=True)   
    return render_template('detail.html', 
                           entity = entity, 
                           width = ww, 
                           imgpath = photopath, 
                           stories = [], 
                           boxes = boxes)

@app.route('/thumbnail', methods=['GET'])
def thumbnail():
    path_id = request.args.get('path_id')
    if path_id is not None:
        try:
            width = session['windowWidth'] / 4.0
            entity = entity_id_map[int(path_id)]
            filename = get_thumbnail(entity.path, entity.en_id, width)
            return send_file(filename, "image/png")
        except Exception as e:
            print("Exception in thumbnail: " + str(e))
    return send_file("static/img/icon_img_red_512px.png", "image/png")

@app.route('/reportsize', methods=['POST'])
def reportsize():
    try:
        session['documentHeight'] = int(request.form.get('document_height'))
        session['documentWidth'] = int(request.form.get('document_width'))
        session['windowHeight'] = int(request.form.get('window_height'))
        session['windowWidth'] = int(request.form.get('window_width'))
    except Exception as e:
        print("In reportsize: " + str(e))

    # app.logger.info("document=(%s,%s), window=(%s,%s)",'
    print("document=(%s,%s), window=(%s,%s)", 
                    session['documentHeight'], session['documentHeight'], 
                    session['windowWidth'], session['windowHeight'])

    return 'OK'
    
@app.route('/search', methods=['GET','POST'])
def search():
    global settings
    global root_entity
    global entity_db_id_map
    if request.method == 'POST':
        data = request.form.get('query')
    else:
        data = request.args.get('query')
    if data is None:
        return render_template('index.html', 
                               entities = root_entity.children, 
                               path_id = -1, 
                               maxfetch = settings.max_photo_fetch)

    db = DataBase()
    photo_ids = []
    for label in str(data).split(sep="and"):
        print("Search token: " + label)
        if label.startswith("face"):
            alias = label.replace("face ", "")
            print("Search alias: " + alias)
            if alias != "":
                some_photo_ids = db.get_photos_with_alias(alias)
        else:
            some_photo_ids = db.get_photos_with_label(label)
        if len(photo_ids) == 0:
            photo_ids.extend(some_photo_ids)
        else:
            intersection = [val for val in photo_ids if val in some_photo_ids]
            photo_ids = intersection
    
    print("Found " + str(len(photo_ids)) + " photos with label " + label)
    entities = []
    for photo_id in photo_ids:
        # entity = root_entity.find_entity_in_tree_with_db_id(photo_id[0])
        if photo_id[0] in entity_db_id_map:
            entity = entity_db_id_map[photo_id[0]]
            if entity is not None:
                entities.append(entity)
        else:
            print("Error: entity " + str(photo_id[0]) + " not found in the map")
    return render_template('index.html', 
                           entities = entities, 
                           path_id = -1, 
                           maxfetch = settings.max_photo_fetch)


@app.route('/updatelabel', methods=['POST'])
def updatelabel():
    photoid = request.form.get("pathid")
    boxid = request.form.get("boxid")
    label = request.form.get("label")
    db = DataBase()
    db.update_box_alias(int(boxid), str(label))
    os.remove(get_filename_cached_photo_labels(photoid))
    return redirect("/detail?path_id=" + photoid)
    
@app.route('/deletecache', methods=['POST'])
def deletecache():
    path = "static/tmp"
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        if os.path.isfile(filepath) and file.startswith("photo"):
            os.remove(filepath)
            
@app.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html')

@app.route('/ai', methods=['GET'])
def ai():
    global settings
    global training_in_progress
    db = DataBase()
    marked_objects = db.get_boxes_with_labels(labelIdx = 6)
    marked_faces = db.get_boxes_with_labels(labelIdx = 7)
    if not training_in_progress:
        can_train = True
        for key in marked_faces.keys():
            if len(marked_faces[key]) < settings.face_cls_training_threshold:
                can_train = False
                break
    else:
        can_train = False
    return render_template('ai.html',
                           marked_objects = marked_objects,
                           marked_faces = marked_faces, 
                           faces_threshold = settings.face_cls_training_threshold, 
                           can_train = can_train,
                           training_in_progress = training_in_progress)

@app.route('/train', methods=['POST'])
def train():
    global proc_pool
    global training_in_progress
    if not training_in_progress:
        training_in_progress = True
        try:
            proc_pool.apply_async(train_classifier, callback = train_classifier_callback)
            proc_pool.close()
        except Exception as e:
            print("Training failure: " + str(e))
    return redirect("/ai")

''' === === === main === === === '''       
def main(args):
    db = DataBase()
    global settings
    settings = Settings.load()
    
    if len(args) >= 2 and args[1] == "-u":
        update_database()
        return
    
    global root_entity
    root_entity = Entity.build_tree(settings.path, "../static/gallery", db, "root")
    
    global proc_pool
    proc_pool = Pool(processes=1)
    
    app.logger.info('Listening on port 8000')
    app.run(host = '0.0.0.0', port=8000, debug=True)
 
if __name__ == "__main__":
    main(sys.argv)
    

