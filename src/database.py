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

# SQLite database has to be in data folder
PATH_TO_DB = "data/gallery.db"

'''
Database manager
'''            
class DataBase:
    def __init__(self):
        self.path_to_db = PATH_TO_DB
        self.db = sqlite3.connect(self.path_to_db)
        self.cursor = self.db.cursor()
        
    def create(self):
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS dirs(id INTEGER PRIMARY KEY, path TEXT unique)''')
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS photos(id INTEGER PRIMARY KEY, path TEXT UNIQUE, hash TEXT)''')
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS boxes(id INTEGER PRIMARY KEY, photoid INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT, alias TEXT, learned INTEGER DEFAULT 0, FOREIGN KEY(photoid) REFERENCES photos(id))''')
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY UNIQUE, val TEXT''')
            self.db.commit()
        except Exception as e:
            print("Exception in create DB: " + str(e))

    '''Get photo id'''
    def get_photo_key(self, photo_path):
        self.cursor.execute('''SELECT id FROM photos WHERE path = ?''', (photo_path,))
        rows = self.cursor.fetchall()
        if len(rows) > 1:
            print("Error: too many ids for the same path!")
        elif len(rows) == 0:
            return -1
        return rows[0][0]

    '''Insert photo path and hash into db and return primary key'''
    def insert_photo(self, photo_path):
        with open(photo_path, 'rb') as fin:
            hs = hashlib.sha256(fin.read()).hexdigest()
            fin.close()
        try:
            self.cursor.execute('''INSERT INTO photos(path, hash) VALUES(?, ?)''', (photo_path, hs))
        except Exception as e:
            print(e)
        self.db.commit()
        return self.get_photo_key(photo_path)
    
    def get_photo(self, photo_id):
        try:
            self.cursor.execute('''SELECT * FROM photos WHERE id = ?''', (photo_id,))
            rows = self.cursor.fetchall()
            if len(rows) > 1:
                print("Error: too many ids for the same path!")
            elif len(rows) == 0:
                return None
            return rows[0]
        except Exception as e:
            print(e)
        return None
    
    def get_photos_with_alias(self, alias):
        try:
            self.cursor.execute('''SELECT DISTINCT photoid FROM boxes WHERE alias = ?''', (alias,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(e)
        return []
    
    def get_photos_with_label(self, label):
        try:
            self.cursor.execute('''SELECT DISTINCT photoid FROM boxes WHERE label = ?''', (label,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(e)
        return []
    
    '''
    labelIdx = 6 - object label (e.g. face, person, car ... )
    labelIdx = 7 - face label
    '''
    def get_boxes_with_labels(self, labelIdx = 6):
        try:
            if labelIdx == 6:
                self.cursor.execute('''SELECT * FROM boxes WHERE label != ""''')
            else:
                self.cursor.execute('''SELECT * FROM boxes WHERE alias != "" AND learned = 2''')
            rows = self.cursor.fetchall()
            data = {}
            for row in rows:
                if row[labelIdx] in data:
                    data[row[labelIdx]].append(row)
                else:
                    data[row[labelIdx]] = [row]
            return data
        except Exception as e:
            print(e)
        return {}
    
    def get_boxes_with_faces(self):
        try:
            self.cursor.execute('''SELECT * FROM boxes WHERE alias = "face" AND learned = 0''')
            return self.cursor.fetchall()
        except Exception as e:
            print(e)
        return []

    def insert_box(self, photo_id, x0, y0, x1, y1, label):
        try:
            self.cursor.execute('''INSERT INTO boxes(photoid, x0, y0, x1, y1, label, alias) VALUES(?, ?, ?, ?, ?, ?, ?)''', (photo_id, x0, y0, x1, y1, label, ""))
            self.db.commit()
        except Exception as e:
            print(e)
            
    def get_boxes(self, photo_id):
        try:
            self.cursor.execute('''SELECT * FROM boxes WHERE photoid = ?''', (photo_id,))
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(e)
        return []
    
    def get_settings(self):
        try:
            self.cursor.execute('''SELECT * FROM settings''')
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(e)
        return []
    
    def update_box_alias(self, boxid, alias):
        try:
            self.cursor.execute('''UPDATE boxes SET alias = ? WHERE id = ?''', (alias, boxid,))
            self.cursor.execute('''UPDATE boxes SET learned = 2 WHERE id = ?''', (boxid,))
            self.db.commit()
        except Exception as e:
            print(e)
                
    def close(self):
        self.db.close()

''' === === === End of DataBase === === === '''