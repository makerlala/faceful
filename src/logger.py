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

class Logger:

    @staticmethod
    def fatal(message):
        message = "\033[91m [FATAL] " + message + "\033[0m"
        print(message)
        sys.exit(1)

    @staticmethod
    def error(message):
        message = "\033[91m [ERROR] " + message + "\033[0m"
        print(message)

    @staticmethod
    def warning(message):
        message = "\033[93m [WARNING] " + message + "\033[0m"
        print(message)

    @staticmethod
    def debug(message):
        message = "\033[94m [DEBUG] " + message + "\033[0m"
        print(message)

    @staticmethod
    def info(message):
        message = "\033[92m [INFO] " + message + "\033[0m"
        print(message)

