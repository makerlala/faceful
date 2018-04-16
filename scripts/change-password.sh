#!/bin/bash
#
# Copyright (c) 2018 - Dumi Loghin (dumi@makerlala.com)
#
# This file is part of faceful - an open source smart photo gallery with 
# object and face recognition.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
PYTHON="python3"
PIP="pip3"
VENV="env3"
PYTHON_VERSION=`$PYTHON -V 2>&1`

# Python virtual environment
cd ../..
if [ "$PYTHON" == "python3" ]; then
	$PYTHON -m venv $VENV
else
	virtualenv $VENV
fi
source $VENV/bin/activate
$PIP install --upgrade flask

cd faceful
$PYTHON src/change-password.py


