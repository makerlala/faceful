# faceful
A Photo Gallery with Face Recognition

## Get started

This gallery has two main components, (i) a web server based on Python Flask and (ii) an AI image processing server running on Tensorflow. 
These two components can run on the same machine or on different machines. If you run everything on a single machine, follow all these steps.
If you run on two separate machines, run the install step on the machine that acts as object and face detection server. We recommend that you 
use a machine with GPU (e.g. a server with GPU or an Nvidia Jetson TX1/TX2) for this.

### Install

Create a folder where all related code and projects will be downloaded, then clone this repository. For example, in your home directory:
```
cd ~
mkdir my-smart-gallery
git clone https://github.com/makerlala/faceful.git
```
Copy the settings template and add the path to your photos and the path to this project folder.
```
cd faceful/scripts
cp cp settings.conf.template settings.conf
vim settings.conf
```

Create a models folder and download pre-trained model 20170512-110547 from facenet repository, https://github.com/davidsandberg/facenet. 
Unzip this model in models folder. Next, you can run install.sh from scripts folder. This will take care of the installation. If you do not 
have a GPU, just install with CPU-only support:
```
./install.sgh -cpu
```
else
```
./install.sh
```

### Install on Jetson TX1 / TX2

On Nvidia Jetson machines, you need to install OpenCV, scipy, tensorflow for aarch64. For OpenCV, we have compile it from sources and provide 
the share library. Download this library from https://drive.google.com/open?id=1s6mlQbWeG1pSyh-Qrr6ukFm7WbTtkc9z and place it in the tensorflow
folder. The install script will place it in the site-packages folder. You can find a guide on how to compile OpenCV on Jetson here: https://jkjung-avt.github.io/opencv3-on-tx2.

For scipy, install it using:
```
sudo apt-get install python-scipy
ln -s /usr/lib/python2.7/dist-packages/scipy scipy
cd my-smart-gallery/env3/lib/python2.7/site-packages
sudo apt-get install python3-scipy
cd my-smart-gallery/env3/lib/python3.5/site-packages
ln -s /usr/lib/python3/dist-packages/scipy scipy
```

We have compiled tensorflow 1.5.0rc1 for Jetson and made it available here:
https://drive.google.com/open?id=16apjnV-SKOepWou8Jcre-LjjEVMDQ_au

Download the file, place it in the tensorflow folder and run install.sh.

### Folder structure

After installation, your folder structure should be:
```
my-smart-gallery
	faceful
		data
		scripts
		src
		static
			assests
			gallery -> <your photos path>
			img
		templates
	facenet
	models
		20170512-110547
		lfw
		ssd_mobilenet_v1_coco_2017_11_17
	tensorflow
		models
	tensorflow-face-detection
```

### Run webserver
If you run both components on the same machine, just start the Flask server. Else, on the machine that acts like webserver, clone this repository, set the path to your photos, and set the facedet server IP and port. Then start Flask server:
```
./start-flask.sh
```

If everything is ok, open a browser and go to http://localhost:8000 or http://<webserver>:8000

Enjoy!

## Credits
Webpage template Radius is created by TEMPLATED (https://templated.co/radius) and released for free under the Creative Commons Attribution 3.0 license (https://templated.co/license)

Parts of the code are inspired or copied from facenet project  by David Sandberg which is licensed under MIT licence (see https://github.com/davidsandberg/facenet).
