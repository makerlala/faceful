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
FACENET_MODEL="20170512-110547/20170512-110547.pb"
FACENET_URL="https://drive.google.com/uc?id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk&export=download"
FACENET_LFW_URL="http://vis-www.cs.umass.edu/lfw/lfw.tgz"
FACENET_LFW_ARCHIVE="lfw.tgz"
FACENET_LFW_PKL="lfw_classifier-20170512-110547.pkl"
OBJECTDET_MODEL="ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
OBJECTDET_URL="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz"
OBJECTDET_ARCHIVE="ssd_mobilenet_v1_coco_2017_11_17.tar.gz"

PYTHON="python3"
PIP="pip3"
VENV="env3"
PYTHON_VERSION=`$PYTHON -V 2>&1`

# by default use GPU
USE_GPU=1
TENSORFLOW_MODULE="tensorflow-gpu"

echo "Using $PYTHON_VERSION"

if [ $# -gt 0 ] && [ $1 == "-h" ]; then
	echo "Usage: $0 [-cpu (CPU-only) | -h (help)]"
	exit
fi
if [ $# -gt 0 ] && [ $1 == "-cpu" ]; then
	USE_GPU=0
	TENSORFLOW_MODULE="tensorflow"
	echo "Using tensorflow CPU only"
fi

# Configure settings
if ! [ -f settings.conf ]; then
	cp settings.conf.template settings.conf
	echo "Please add the path to your photos in 'settings.conf' and run this script again!"
	exit
else
	PHOTOS_PATH=`cat settings.conf | grep "photos path =" | tr -d ' ' | cut -d '=' -f 2`
	echo "Path to your photos: $PHOTOS_PATH"
	if ! [ -d "$PHOTOS_PATH" ]; then
		echo "Photos path is not a folder. Please reconfigure your path in 'settings.conf' and run this script again!"
		exit
	fi
	cd ../static
	rm -f gallery
	ln -s $PHOTOS_PATH gallery
	cd ../scripts
fi

cd ..
mkdir data

# go to one level above facegaph root
cd ..

# Install facenet
if ! [  -d facenet ]; then
git clone https://github.com/davidsandberg/facenet.git
fi

# Install tensorflow models
if ! [ -d tensorflow/models ]; then
mkdir -p tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cd ../../..
fi

# Install facedetection model
if ! [ -d tensorflow-face-detection ]; then
git clone https://github.com/yeephycho/tensorflow-face-detection.git
fi

# Install models
mkdir -p models
if ! [ -f models/$FACENET_MODEL ]; then
	echo "Download and unzip facenet model in models folder, and run this script again!"
	echo "Facenet model URL: $FACENET_URL"
	exit
fi
if ! [ -f models/$OBJECTDET_MODEL ]; then
	cd models
	if ! [ -f $OBJECTDET_ARCHIVE ]; then
		wget $OBJECTDET_URL
	fi
	tar xf $OBJECTDET_ARCHIVE
	if ! [ -f $OBJECTDET_MODEL ]; then
		echo "Download and unzip object detection model in models folder, and run this script again!"
		echo "Object detection model URL: $OBJECTDET_URL"
		exit
	fi
	cd ..
fi

mkdir -p models/lfw/raw
cd models
if ! [ -f $FACENET_LFW_ARCHIVE ]; then
	wget $FACENET_LFW_URL
fi
tar xf $FACENET_LFW_ARCHIVE -C lfw/raw --strip-components=1

# go to one level above facegaph root
cd ..

# Python virtual environment
if [ "$PYTHON" == "python3" ]; then
	$PYTHON -m venv $VENV
else
	virtualenv $VENV
fi
source $VENV/bin/activate
$PIP install --upgrade html5lib scipy Pillow absl-py sklearn opencv-python $TENSORFLOW_MODULE

# Create classifier environment
cd facenet
export PYTHONPATH="src"
if ! [ -d ../models/lfw/lfw_mtcnnpy_160 ]; then
	for N in {1..4}; do 
		python src/align/align_dataset_mtcnn.py ../models/lfw/raw ../models/lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 &
	done
	wait
fi
# Train LFW classifier
if ! [ -f ../models/$FACENET_LFW_PKL ]; then
$PYTHON src/classifier.py TRAIN ../models/lfw/lfw_mtcnnpy_160 ../models/$FACENET_MODEL ../models/$FACENET_LFW_PKL --batch_size 1000
fi
# Test
echo ""
echo "Testing LFW..."
# $PYTHON src/validate_on_lfw.py ../models/lfw/lfw_mtcnnpy_160 ../models/20170512-110547
cd ..
deactivate

