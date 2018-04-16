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
import getpass
from werkzeug.security import generate_password_hash, check_password_hash

from database import DataBase


def main(args):
    print("Change password for 'faceful' user admin.")
    passwd1 = getpass.getpass()
    print("Repeat password.")
    passwd2 = getpass.getpass()
    if passwd1 != passwd2:
        print("Passwords do not match. Try again.")
        return 1

    if not passwd1 or len(passwd1) < 8:
        print("Invalid password. Password must be at least 8 characters long.")
        return 2

    db = DataBase()
    db.set_user("admin", generate_password_hash(passwd1))

    print("Done.")

    return 0


if __name__ == "__main__":
    main(sys.argv)