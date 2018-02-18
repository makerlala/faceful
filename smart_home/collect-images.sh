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
CONFIG_FILE="camera-config.csv"

if ! [ -f $CONFIG_FILE ]; then
	echo "No config file. Exiting."
	exit
fi
IFS=$'\n'
for LINE in $(cat $CONFIG_FILE); do
	if [ ${LINE:0:1} == "#" ]; then
		continue
	fi
	NAME=`echo $LINE | cut -d ';' -f 1`
	TYPE=`echo $LINE | cut -d ';' -f 2`
	XHOST=`echo $LINE | cut -d ';' -f 3`
	XPATH=`echo $LINE | cut -d ';' -f 4`
    USER=`echo $LINE | cut -d ';' -f 5`
    PASS=`echo $LINE | cut -d ';' -f 6`
    # echo $NAME $TYPE $XHOST $XPATH $USER $PASS
	case $TYPE in
	"jetson")
		ssh $XHOST "bash $XPATH/get-one-shot-jetson.sh"
		scp $XHOST:one-shot.jpg $NAME.jpg
		# date +%y%m%d-%H%M%S > $NAME.txt
        date +%s > $NAME.txt
	    ;;
    "ipcam")
        wget --user=$USER --password=$PASS $XHOST$XPATH -O $NAME.jpg > /dev/null 2>&1
        # date +%y%m%d-%H%M%S > $NAME.txt
        date +%s > $NAME.txt
        ;;
	esac    
done
