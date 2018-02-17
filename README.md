# faceful
A Photo Gallery with Face Recognition

## Get started

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

Start the Flask server:
```
./start-flask.sh
```

If everything is ok, open a browser and go to "localhost:8000"

Enjoy!

## Credits
Webpage template Radius is created by TEMPLATED (emplated.co @templatedco) and released for free under the Creative Commons Attribution 3.0 license (templated.co/license)

Parts of the code are inspired or copied from facenet project  by David Sandberg which is licensed under MIT licence (see https://github.com/davidsandberg/facenet).
