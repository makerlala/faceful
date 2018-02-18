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
# from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl
import json
import sys
import csv
import os.path
import time

users_full = {}
users_nick = {}

facedet_all_file = None
facedet_front_file = None

class Handler(BaseHTTPRequestHandler):
    def _set_headers(self, code):
        self.send_response(code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def reply(self, message):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        message = message.decode()
        message = '{speech:"' + message + '",displayText:"' + message + '"}'
        self.wfile.write(message.encode())

    def do_GET(self):
        self._set_headers(404)        

    def do_HEAD(self):
        self._set_headers(404)
        
    def do_POST(self):
        time.sleep(3)
        try:
            msg = "Sorry, I can't help you with that."
            json_data = self.rfile.read(int(self.headers['Content-Length']))            
            data = json.loads(json_data.decode())
            print(data)           
            action = data['result']['action']
            if action == "get_user_location":
                first_name = data['result']['parameters']['given-name']
                last_name = data['result']['parameters']['last-name']
                full_name = first_name + ' ' + last_name
                if full_name in users_full or first_name in users_nick:
                    msg = first_name + " is doing great"
                else:
                    msg = "No such user"
            elif action == "get_user_front":
                if not os.path.isfile(facedet_front_file):
                    msg = "I can see nobody in front of me. Am I blind?"
                else:
                    with open(facedet_front_file, 'rt') as fff:
                        user = fff.read()
                        fff.close
                    os.remove(facedet_front_file)
                    msg = "I can see you, " + user
            elif action == "get_all_users":
                if not os.path.isfile(facedet_all_file):
                    msg = "I can see nobody in the house."
                else:
                    with open(facedet_all_file, 'rt') as fff:
                        msg = fff.read()
                        fff.close                    
            else:
                msg = "I don't know how to help with that"
            self.reply(msg.encode())    
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(str(e))
            print(e)
        
def run(settings_file, users_file, port = 8443):
    global facedet_all_file
    global facedet_front_file
    global users_full
    global users_nick
    
    # load settings
    cert_file = ""
    cert_priv_key_file = ""
    with open(settings_file, 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[0].startswith('#'):
                continue
            elif row[0] == "certificate file":
                cert_file = row[1]
            elif row[0] == "certificate private key file":
                cert_priv_key_file = row[1]
            elif row[0] == "facedet file":
                facedet_all_file = row[1]
            elif row[0] == "facedet front file":
                facedet_front_file = row[1]
                
    if cert_file == "" or cert_priv_key_file == "" or not os.path.isfile(cert_file) or not os.path.isfile(cert_priv_key_file):
        print("Certificate files not configured properly!")
        return
    
    # load users
    with open(users_file, 'rt') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[0].startswith('#'):
                continue
            users_full[row[0]] = [row[0], row[1]]
            users_nick[row[1]] = [row[0]]
            print(users_nick[row[1]])

    httpd = HTTPServer(('', port), Handler)
    httpd.socket = ssl.wrap_socket(httpd.socket, keyfile=cert_priv_key_file, certfile=cert_file, server_side=True)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " <settings_file> <users_file>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
