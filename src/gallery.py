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
from werkzeug.security import check_password_hash

# a pool of async processes
proc_pool = None
# Object detection in progress
detection_in_progress = False
# Face classifier training in progress
training_in_progress = False

from networking import Connection
from settings import Settings
from database import DataBase
from logger import Logger

settings = None

'''
Entity: folder, image, video
- arranged as tree for folder navigation
- arranged as map for quick access
- has an id in this app (en_id)
- has a distinct id in the database (db_id)
'''
root_entity = None
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
        global entity_db_id_map

        if not os.path.isdir(path):
            Logger.fatal("Path not found: " + path)
            sys.exit(1)
        root = Entity(vpath, name, root_en_id, -1, True, False)
        root_en_id = root_en_id + 1
        prev_child = None
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            virtual_file_path = vpath + "/" + file
            if os.path.isdir(file_path):
                db_id = -1 * root_en_id
                child = Entity.build_tree(file_path, virtual_file_path, db, file)
            elif file.startswith("."):
                continue
            elif is_video(file) or is_photo(file):
                db_id = db.get_photo_id(file_path)
                child = Entity(virtual_file_path, file, root_en_id, db_id, False, is_video(file))
                if prev_child is not None:
                    child.prev_id = prev_child.en_id
                    prev_child.next_id = child.en_id
                prev_child = child

            else:
                Logger.error("Unknown file: " + file_path)
                continue
            root_en_id = root_en_id + 1
            child.db_id = db_id
            entity_db_id_map[db_id] = child
            root.add_child(child)
            Logger.info("Photo path: " + file_path + " DB id: " + str(db_id))
        return root


def is_photo(f):
    f = str.lower(f)
    return f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".bmp") or f.endswith(".gif")


def is_video(f):
    f = str.lower(f)
    return f.endswith(".mpg") or f.endswith(".avi") or f.endswith(".mp4") or f.endswith(".mov") or f.endswith(".wmv")


def update_database_callback(result):
    global detection_in_progress

    detection_in_progress = False
    Logger.info("Detection (+update database) is done.")


def update_database():
    global settings
    
    db = DataBase()
    db.create()
    
    img_paths = []
    
    for (dirpath, dirnames, filenames) in os.walk(settings.path):
        for f in filenames:
            if is_photo(f):
                img_paths.append(os.path.join(dirpath, f))
    
    n = len(img_paths)
    i = 1.0
    img_ids_to_uplod = []
    img_paths_to_upload = {}
    for img_path in img_paths:
        Logger.info("Update database progress {}".format(100.0*i/n))
        i = i + 1.0      
        img_id = db.get_photo_id(img_path)
        if img_id != -1:
            # Photo is already in the database
            Logger.debug("Photo is already in the database: " + img_path)
            continue

        img_id = db.insert_photo(img_path)
        Logger.debug("Photo inserted in db: {}, {}".format(img_path, img_id))
        img_ids_to_uplod.append(img_id)
        img_paths_to_upload[img_id] = img_path

    if len(img_ids_to_uplod) == 0:
        db.close()
        return

    for i in range(int(len(img_ids_to_uplod)/settings.facedet_batch) + 1):
        start = i * settings.facedet_batch
        end = min(len(img_ids_to_uplod), (i+1) * settings.facedet_batch)
        img_paths = [img_paths_to_upload[id] for id in img_ids_to_uplod[start:end]]
        conn = Connection(settings.facedet_host, settings.facedet_port)
        conn.send_message("objdet")
        conn.upload_images(img_paths, img_ids_to_uplod[start:end])
        ret = conn.get_response()
        Logger.debug("Got " + ret)
        records = ret.split(':')
        for record in records:
            if record == "END":
                break
            tokens = record.split(';')
            if len(tokens) != 6:
                Logger.error("Error in update_database: invalid number of tokens " +
                             len(record) + " in record [" + record + "]")
            else:
                db.insert_box(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[0])
        conn.close()

    db.close()


def train_classifier_callback(result):
    global training_in_progress

    training_in_progress = False
    Logger.info("Training classifier is done.")


def train_classifier():
    global settings

    Logger.info("Start training classifier.")
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
            img = cv2.resize(img[y0:y1,x0:x1,:],
                             (settings.facenet_classifier_image_size, settings.facenet_classifier_image_size))
            try:
                faces[alias].append(cv2.imencode(".png", img)[1].tostring())
            except Exception as e:
                Logger.error("Exception in train_classifier" + str(e))
    
    resp = conn.send_message("train")
    Logger.debug(resp)
    if resp != "OK":
        return
    
    data = "{} {}".format(len(faces.keys()), settings.face_cls_training_threshold)
    Logger.debug("Send: " + data)
    resp = conn.send_message(data)
    Logger.debug(resp)
    if resp != "OK":
        return
    
    for alias in faces.keys():
        resp = conn.send_message(alias)
        Logger.debug(resp)
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


def scale_photo(photo_path, photo_id, scaled_h, scaled_w, cached_photo_path, min_ratio=False):
    img = cv2.imread(photo_path)
    [h, w] = np.asarray(img.shape)[0:2]
    if min_ratio:
        ratio = min(scaled_h/h, scaled_w/w)
    else:
        ratio = float(scaled_w)/float(w)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    cv2.imwrite(cached_photo_path, img)
    return cached_photo_path


def scale_photo_with_boxes(photo_path, photo_id, scaled_h, scaled_w, cached_photo_path, boxes):
    img = cv2.imread(photo_path)
    [h, w] = np.asarray(img.shape)[0:2]
    Logger.debug("1 h {} w {}".format(h, w))
    ratio = min(scaled_h/h, scaled_w/w)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    [h, w] = np.asarray(img.shape)[0:2]
    Logger.debug("2 h {} w {}".format(h, w))
    n = 1
    for box in boxes:
        try:
            x0 = int(float(box[2]) * float(w))
            y0 = int(float(box[3]) * float(h))
            x1 = int(float(box[4]) * float(w))
            y1 = int(float(box[5]) * float(h))
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if box[6] == "face":
                txt = box[7] + " (face " + str(n) + ")"
            else:
                txt = box[6] + " (box " + str(n) + ")"
            n = n + 1
            cv2.putText(img, txt, (x0, y0-10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            Logger.error("Exception in scale_boxes_photo: " + str(e))
    cv2.imwrite(cached_photo_path, img)
    return cached_photo_path


def get_filename_cached_photo(photo_id):
    return "static/tmp/photo_" + str(photo_id) + ".png"


def get_filename_cached_photo_labels(photo_id):
    return "static/tmp/photo_labels_" + str(photo_id) + ".png"
    

def get_thumbnail(photo_path, photo_id, scaled_w):
    filename = "static/tmp/thumbnail_" + str(photo_id) + ".png"
    if os.path.isfile(filename):
        return filename
    return scale_photo(photo_path, photo_id, 0, scaled_w, filename)


def get_photo(photo_path, photo_id, window_h, window_w, boxes, with_labels=False):
    if with_labels:
        filename = get_filename_cached_photo_labels(photo_id)
    else:
        filename = get_filename_cached_photo(photo_id)
    if os.path.isfile(filename):
        return filename
    if with_labels:
        return scale_photo_with_boxes(photo_path, photo_id, 0.9 * window_h, 0.6 * window_w, filename, boxes)
    else:
        return scale_photo(photo_path, photo_id, 0.9 * window_h, 0.6 * window_w, filename, min_ratio=True)


''' ============================= 
    === === === Flask === === ===
    ============================= '''

app = Flask("My Smart Gallery")
app.secret_key = 'uaYX3T283RGDW384T1EGVas1363'


@app.route('/', methods=['GET'])
def index():
    if not check_session_login():
        return redirect("/login")

    global settings
    global root_entity
    global entity_db_id_map

    path_id = request.args.get('path_id')
    if path_id is not None:
        path_id = int(path_id)
        entity = entity_db_id_map[path_id]
        if entity is not None:
            session['lastIndex' + str(path_id)] = settings.max_photo_fetch
            return render_template('index.html', 
                                   entities=entity.children,
                                   path_id=path_id,
                                   maxfetch=settings.max_photo_fetch)
            
    return render_template('index.html', 
                           entities=root_entity.children,
                           path_id=-1,
                           maxfetch=settings.max_photo_fetch)


@app.route('/getmorephotos', methods=['POST'])
def getmorephotos():
    if not check_session_login():
        return redirect("/login")

    global settings
    global root_entity
    global entity_db_id_map

    path_id = request.form.get('path_id')
    if path_id is not None:
        path_id = int(path_id)
        entity = entity_db_id_map[path_id]
        if entity is None:
            return '';
        last_index_key = 'last_index' + str(path_id)
        if last_index_key in session:
            last_index = session[last_index_key]
        else:
            last_index = 0
        data = ""
        next_index = min(last_index + settings.max_photo_fetch, len(entity.children))
        session[last_index_key] = next_index
        for i in range(last_index, next_index):
            child = entity.children[i]
            if child.is_video:
                data = data + '<div class="image fit"><a href="/detail?path_id=' + str(child.db_id) + \
                       '"><img src="static/img/icon_video_red_512px.png" alt="' + child.name + '" /></a</div>'
            else:
                data = data + '<div class="image fit"><a href="/detail?path_id=' + str(child.db_id) + \
                       '"><img src="/thumbnail?path_id=' + str(child.db_id) + '&w=400" alt="' + child.name + \
                       '" /></a></div>'
        return data
            

@app.route('/detail', methods=['GET'])
def detail():
    if not check_session_login():
        return redirect("/login")

    path_id = request.args.get('path_id')
    if path_id is None:
        Logger.error("Error in /detail: " + path_id + " not found")
        return "Not found!"
    
    path_id = int(path_id)
    entity = entity_db_id_map[path_id]
    if entity is None:
        Logger.error("Error in /detail: " + path_id + " not found")
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
                               entity=entity,
                               width=ww,
                               imgpath="",
                               stories=[],
                               boxes=[])
    
    db = DataBase()
    boxes = db.get_boxes(entity.db_id)
    db.close()
    Logger.debug("Detail boxes: " + str(boxes))
    photopath = get_photo(entity.path, entity.db_id, wh, ww, boxes, with_labels=True)
    return render_template('detail.html', 
                           entity=entity,
                           width=ww,
                           imgpath=photopath,
                           stories=[],
                           boxes=boxes)


@app.route('/thumbnail', methods=['GET'])
def thumbnail():
    if not check_session_login():
        return redirect("/login")

    global entity_db_id_map
    path_id = request.args.get('path_id')
    if path_id is not None:
        try:
            width = session['windowWidth'] / 4.0
            entity = entity_db_id_map[int(path_id)]
            filename = get_thumbnail(entity.path, entity.db_id, width)
            return send_file(filename, "image/png")
        except Exception as e:
            Logger.error("Exception in thumbnail: " + str(e))
    return send_file("static/img/icon_img_red_512px.png", "image/png")


# This is called by the client to report browser dimensions
@app.route('/reportsize', methods=['POST'])
def reportsize():
    if not check_session_login():
        return redirect("/login")

    try:
        session['documentHeight'] = int(request.form.get('document_height'))
        session['documentWidth'] = int(request.form.get('document_width'))
        session['windowHeight'] = int(request.form.get('window_height'))
        session['windowWidth'] = int(request.form.get('window_width'))
    except Exception as e:
        Logger.error("In reportsize: " + str(e))

    Logger.debug("document=({},{}), window=({},{})".format(
                 session['documentHeight'], session['documentHeight'],
                 session['windowWidth'], session['windowHeight']))

    return 'OK'
    

@app.route('/search', methods=['GET','POST'])
def search():
    global settings
    global root_entity
    global entity_db_id_map

    if not check_session_login():
        return redirect("/login")

    if request.method == 'POST':
        data = request.form.get('query')
    else:
        data = request.args.get('query')
    if data is None:
        return render_template('index.html', 
                               entities=root_entity.children,
                               path_id=-1,
                               maxfetch=settings.max_photo_fetch)

    db = DataBase()
    photo_ids = []
    for label in str(data).split(sep="and"):
        Logger.debug("Search token: " + label)
        if label.startswith("face") and len(label) > 4:
            alias = label.replace("face ", "")
            Logger.debug("Search alias: " + alias)
            if alias != "":
                some_photo_ids = db.get_photos_with_alias(alias)
        else:
            some_photo_ids = db.get_photos_with_label(label)
        if len(photo_ids) == 0:
            photo_ids.extend(some_photo_ids)
        else:
            intersection = [val for val in photo_ids if val in some_photo_ids]
            photo_ids = intersection

    Logger.debug("Found " + str(len(photo_ids)) + " photos with label " + label)
    entities = []
    for photo_id in photo_ids:
        # entity = root_entity.find_entity_in_tree_with_db_id(photo_id[0])
        if photo_id[0] in entity_db_id_map:
            entity = entity_db_id_map[photo_id[0]]
            if entity is not None:
                entities.append(entity)
        else:
            Logger.error("Entity " + str(photo_id[0]) + " not found in the map")
    return render_template('index.html', 
                           entities=entities,
                           path_id=-1,
                           maxfetch=settings.max_photo_fetch)


@app.route('/updatelabel', methods=['POST'])
def updatelabel():
    if not check_session_login():
        return redirect("/login")

    photoid = request.form.get("pathid")
    boxid = request.form.get("boxid")
    label = request.form.get("label")
    db = DataBase()
    db.update_box_alias(int(boxid), str(label))
    os.remove(get_filename_cached_photo_labels(photoid))
    return redirect("/detail?path_id=" + photoid)
    

@app.route('/deletecache', methods=['POST'])
def deletecache():
    if not check_session_login():
        return redirect("/login")

    path = "static/tmp"
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        if os.path.isfile(filepath) and file.startswith("photo"):
            os.remove(filepath)
            

@app.route('/settings', methods=['GET'])
def settings():
    if not check_session_login():
        return redirect("/login")

    # TODO
    return render_template('settings.html')


@app.route('/ai', methods=['GET'])
def ai():
    global settings
    global training_in_progress

    if not check_session_login():
        return redirect("/login")

    db = DataBase()
    marked_objects = db.get_boxes_with_labels(label_index = 6)
    marked_faces = db.get_boxes_with_labels(label_index = 7)

    if not training_in_progress:
        can_train = True
        for key in marked_faces.keys():
            if len(marked_faces[key]) < settings.face_cls_training_threshold:
                can_train = False
                break
    else:
        can_train = False

    return render_template('ai.html',
                           marked_objects=marked_objects,
                           marked_faces=marked_faces,
                           faces_threshold=settings.face_cls_training_threshold,
                           can_train=can_train,
                           training_in_progress=training_in_progress,
                           detection_in_progress=detection_in_progress)


@app.route('/train', methods=['POST'])
def train():
    global proc_pool
    global training_in_progress

    if not check_session_login():
        return redirect("/login")

    if not training_in_progress:
        training_in_progress = True
        try:
            proc_pool.apply_async(train_classifier, callback=train_classifier_callback)
            # proc_pool.close()
        except Exception as e:
            training_in_progress = False
            Logger.error("Training failure: " + str(e))

    return redirect("/ai")


@app.route('/detect', methods=['POST'])
def detect():
    global proc_pool
    global detection_in_progress

    if not check_session_login():
        return redirect("/login")

    if not detection_in_progress:
        detection_in_progress = True
        try:
            proc_pool.apply_async(update_database, callback=update_database_callback)
            # proc_pool.close()
        except Exception as e:
            detection_in_progress = False
            Logger.error("Failure in running detection: " + str(e))

    return redirect("/ai")


'''Check login credentials against DB'''
def check_login(name, passwd):
    db = DataBase()
    db_passwd = db.get_password(name)
    if not db_passwd:
        return False
    return check_password_hash(db_passwd, passwd)


'''Check login credentials from session'''
def check_session_login():
    if "un" in session:
        name = session["un"]
    else:
        name = None
    if "pw" in session:
        passwd = session["pw"]
    else:
        passwd = None
    if not name or not passwd:
        return False
    return check_login(name, passwd)


@app.route('/login', methods=['GET'])
def show_login():
    if check_session_login():
        return redirect("/")
    return render_template('login.html', message="")


@app.route('/login', methods=['POST'])
def process_login():
    name = request.form.get('name')
    passwd = request.form.get('passwd')
    if not name or not passwd:
        return render_template('login.html', message="Empty user name or password")

    if check_login(name, passwd):
        session["un"] = name
        session["pw"] = passwd
        return redirect("/")

    return render_template('login.html', message="Invalid user name or password")


def main(args):
    db = DataBase()
    db.create()

    global settings
    settings = Settings.load()
    
    if len(args) >= 2 and args[1] == "-u":
        update_database()
        return
    
    global root_entity
    root_entity = Entity.build_tree(settings.path, "static/gallery", db, "root")
    
    if not os.path.isdir("static/tmp"):
        os.mkdir("static/tmp")
    
    global proc_pool
    proc_pool = Pool(processes=4)
    
    app.logger.info('Listening on port 8000')
    app.run(host = '0.0.0.0', port=8000, debug=True)


if __name__ == "__main__":
    main(sys.argv)
    

