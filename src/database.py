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

import sqlite3
import hashlib
from logger import Logger

# SQLite database has to be in 'data' folder
PATH_TO_DB = "data/gallery.db"


class DataBase:
    def __init__(self):
        self.path_to_db = PATH_TO_DB
        self.db = sqlite3.connect(self.path_to_db)
        self.cursor = self.db.cursor()

    '''Create all tables'''
    def create(self):
        # Table for users (only one user)
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, 
            passwd TEXT NOT NULL)''')
        except Exception as e:
            Logger.fatal("Exception in creating users db: " + str(e))

        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS dirs(id INTEGER PRIMARY KEY, path TEXT unique)''')
        except Exception as e:
            Logger.fatal("Exception in creating dirs db: " + str(e))

        # Table for photos
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS photos(id INTEGER PRIMARY KEY, 
            path TEXT UNIQUE, hash TEXT)''')
        except Exception as e:
            Logger.fatal("Exception in creating photos db: " + str(e))

        # Table for objects (marked by boxes)
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS boxes(id INTEGER PRIMARY KEY, photoid INTEGER, 
            x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT, alias TEXT, learned INTEGER DEFAULT 0, 
            FOREIGN KEY(photoid) REFERENCES photos(id))''')
        except Exception as e:
            Logger.fatal("Exception in creating boxes db: " + str(e))

        # Table for stories
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS stories(id INTEGER PRIMARY KEY, photoid INTEGER,
            story TEXT, FOREIGN KEY(photoid) REFERENCES photos(id))''')
        except Exception as e:
            Logger.fatal("Exception in creating stories db: " + str(e))

        # Table for settings
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS settings(key PRIMARY KEY UNIQUE, val TEXT)''')
        except Exception as e:
            Logger.fatal("Exception in creating settings db: " + str(e))

        try:
            self.db.commit()
        except Exception as e:
            Logger.fatal("Exception in db commit: " + str(e))

    '''Get user's hashed password'''
    def get_password(self, name):
        try:
            self.cursor.execute('''SELECT * FROM users WHERE name = ?''', (name,))
            rows = self.cursor.fetchall()
            if len(rows) < 1:
                return None
            else:
                return rows[0][2]
        except Exception as e:
            Logger.error("Exception in db select: " + str(e))
        return None

    '''Add new user or update existing user'''
    def set_user(self, name, passwd):
        if self.get_password(name):
            # user exists, just update password
            try:
                self.cursor.execute('''UPDATE users SET passwd = ? WHERE name = ?''', (name, passwd,))
                self.db.commit()
            except Exception as e:
                Logger.error("Exception in updating user's password: " + str(e))
                return False
        else:
            try:
                self.cursor.execute('''INSERT INTO users(name, passwd) VALUES(?, ?)''', (name, passwd))
                self.db.commit()
            except Exception as e:
                Logger.error("Exception in inserting new user: " + str(e))
                return False
        return True

    '''Get photo id'''
    def get_photo_id(self, photo_path):
        try:
            self.cursor.execute('''SELECT id FROM photos WHERE path = ?''', (photo_path,))
            rows = self.cursor.fetchall()
            if len(rows) > 1:
                Logger.error("Too many ids for the same path.")
            elif len(rows) == 0:
                return -1
        except Exception as e:
            Logger.error("Exception in get_photo_id: " + str(e))
            return -1
        return int(rows[0][0])

    '''Insert photo path and hash into db and return primary key'''
    def insert_photo(self, photo_path):
        try:
            with open(photo_path, 'rb') as fin:
                hs = hashlib.sha256(fin.read()).hexdigest()
                fin.close()
            self.cursor.execute('''INSERT INTO photos(path, hash) VALUES(?, ?)''', (photo_path, hs))
            self.db.commit()
        except Exception as e:
            Logger.error("Exception in insert_photo: " + str(e))
        return self.get_photo_id(photo_path)
    
    def get_photo(self, photo_id):
        try:
            self.cursor.execute('''SELECT * FROM photos WHERE id = ?''', (photo_id,))
            rows = self.cursor.fetchall()
            if len(rows) > 1:
                Logger.error("Too many ids for the same path!")
            elif len(rows) == 0:
                return None
            return rows[0]
        except Exception as e:
            Logger.error(e)
        return None
    
    def get_photos_with_alias(self, alias):
        try:
            self.cursor.execute('''SELECT DISTINCT photoid FROM boxes WHERE alias = ?''', (alias,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            Logger.error(e)
        return []
    
    def get_photos_with_label(self, label):
        try:
            self.cursor.execute('''SELECT DISTINCT photoid FROM boxes WHERE label = ?''', (label,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            Logger.error(e)
        return []
    
    '''
    labelIdx = 6 - object label (e.g. face, person, car ... )
    labelIdx = 7 - face label
    '''
    def get_boxes_with_labels(self, label_index=6):
        try:
            if label_index == 6:
                self.cursor.execute('''SELECT * FROM boxes WHERE label != ""''')
            else:
                self.cursor.execute('''SELECT * FROM boxes WHERE alias != "" AND learned = 2''')
            rows = self.cursor.fetchall()
            data = {}
            for row in rows:
                if row[label_index] in data:
                    data[row[label_index]].append(row)
                else:
                    data[row[label_index]] = [row]
            return data
        except Exception as e:
            Logger.error(e)
        return {}
    
    def get_boxes_with_faces(self):
        try:
            self.cursor.execute('''SELECT * FROM boxes WHERE alias = "face" AND learned = 0''')
            return self.cursor.fetchall()
        except Exception as e:
            Logger.error(e)
        return []

    def insert_box(self, photo_id, x0, y0, x1, y1, label):
        try:
            self.cursor.execute('''INSERT INTO boxes(photoid, x0, y0, x1, y1, label, alias) 
            VALUES(?, ?, ?, ?, ?, ?, ?)''', (photo_id, x0, y0, x1, y1, label, ""))
            self.db.commit()
        except Exception as e:
            Logger.error(e)
            
    def get_boxes(self, photo_id):
        try:
            self.cursor.execute('''SELECT * FROM boxes WHERE photoid = ?''', (photo_id,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            Logger.error(e)
        return []
    
    def get_settings(self):
        try:
            self.cursor.execute('''SELECT * FROM settings''')
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            Logger.error(e)
        return []
    
    def update_box_alias(self, boxid, alias):
        try:
            self.cursor.execute('''UPDATE boxes SET alias = ? WHERE id = ?''', (alias, boxid,))
            self.cursor.execute('''UPDATE boxes SET learned = 2 WHERE id = ?''', (boxid,))
            self.db.commit()
        except Exception as e:
            Logger.error(e)

    def get_story(self, photo_id):
        try:
            self.cursor.execute('''SELECT * FROM stories WHERE photoid = ?''', (photo_id,))
            rows = self.cursor.fetchall()
            return rows[0]
        except Exception as e:
            Logger.error(e)
        return None

    def add_story(self, photo_id, story_text):
        story = self.get_story(photo_id)
        if story:
            try:
                self.cursor.execute('''UPDATE stories SET story = ? WHERE photoid = ?''', (story_text, photo_id,))
                self.db.commit()
            except Exception as e:
                Logger.error(e)
        else:
            try:
                self.cursor.execute('''INSERT INTO stories(photoid, story) VALUES(?,?)''', (photo_id, story_text))
                self.db.commit()
            except Exception as e:
                Logger.error(e)
            story = self.get_story(photo_id)
        # return id
        return story[0]

    def close(self):
        self.db.close()


''' === === === End of DataBase === === === '''
