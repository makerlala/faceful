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
import socket
import os

CHUNK = 1024


class Connection:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host,port))
        self.sock.settimeout(600)
    
    def close(self):
        self.sock.close()

    def send_message(self, message):
        self.sock.send(str.encode(message))
        return self.sock.recv(CHUNK).decode()
    
    '''Upload images through socket'''
    def upload_images(self, images, images_id, in_mem=False):
        data_images = ""
        total_images = len(images)
        if in_mem:
            for i in range(total_images):
                data_images = data_images + "png," + str(len(images[i])) + "," + str(images_id[i]) + ";"
        else:
            for i in range(total_images):
                tokens = images[i].split(".")
                if len(tokens) > 0:
                    img_type = tokens[len(tokens)-1]
                else:
                    img_type = "na"
                img_size = os.path.getsize(images[i])
                data_images = data_images + img_type + "," + str(img_size) + "," + str(images_id[i]) + ";"
                     
        try:
            # send images information
            self.sock.send(str.encode(data_images))
            data = self.sock.recv(CHUNK).decode()
            if data != "OK":
                print("In upload_images we received: " + data)
                return
            
            # send images one by one
            for image in images:
                if in_mem:
                    data = image
                else:
                    with open(image, 'rb') as f:
                        data = f.read()
                        f.close()
                        
                print("Image size [bytes] in memory: " + str(len(data)))
                if len(data) <= CHUNK:
                    self.sock.send(data)
                else:
                    start = 0
                    for i in range(len(data) // CHUNK):
                        start = i * CHUNK
                        end = start + CHUNK
                        nbytes = self.sock.send(data[start:end])
                        # print("Bytes sent on socket: " + str(nbytes))
                    start = start + CHUNK
                    end = len(data)
                    if (end >= start):
                        nbytes = self.sock.send(data[start:end])
                        # print("Bytes sent on socket: " + str(nbytes))  
                data = self.sock.recv(CHUNK).decode()
                if data != "OK":
                    print("In upload_images we received: " + data)
                    return
        except Exception as e:
            print(e)            
        print("Uploading images done.")
    
    def get_response(self):
        response = ""
        while True:
            data = self.sock.recv(CHUNK).decode()
            response = response + data
            if 'END' in data:
                break
        return response
    
    '''
    Download images through socket
    Client should first send a list of images of the form:
    <img_type0,size0>;<img_type1,size1>;...
    '''
    @staticmethod
    def download_images(conn, in_mem = False, close_conn = True):
        img_data = []
        img_paths = []
        img_ids = []
    
        try:
            data = conn.recv(2048)
            data_split = data.decode().split(";")
            conn.send(str.encode("OK"))
            n = 0
            for token in data_split:
                if len(token) < 1:
                    continue
                image = token.split(",")
                print("Getting a " + image[0].lower() + " image of size " + image[1])        
                img_name = "img" + str(n) + "." + image[0].lower()
                img_size = int(image[1])
                img_ids.append(image[2])
                img_buf = b''
                curr_size = 0
                if img_size <= CHUNK:
                    img_buf = conn.recv(img_size)
                else:
                    while curr_size < img_size:
                        data_chunk = conn.recv(CHUNK)
                        #if len(data_chunk) != CHUNK:
                        #    print("Warning in receive: expected " + str(CHUNK) + " bytes, received " + str(len(data_chunk)) + " bytes")            
                        curr_size = curr_size + len(data_chunk)                       
                        img_buf = img_buf + data_chunk
                if in_mem:
                    img_data.append(img_buf)
                else:
                    with open(img_name, 'wb') as f:
                        f.write(img_buf)
                        f.close()
                    img_paths.append(img_name)
                print("Got an image of size: " + str(len(img_buf)))
                conn.send(str.encode("OK"))
        
            if close_conn:
                conn.close()
                
        except Exception as e:
            print(e)
    
        return img_data, img_paths, img_ids
