# Setting up the Jetson Nano with the necessary libraries
**Contents:**
1. [**Getting Started with the Jetson Nano**](#getting-started-with-the-jetson-nano)
2. [**Configuring Nano for SSH**](#configuring-nano-for-ssh)
3. [**Basic Setup**](#basic-setup) - [Python](#installing-python), [Others](#installing-others)
4. [**Python vs Python3**](#python-vs-python3)
5. [**Installing Basic Python Packages**](#installing-basic-python-packages)
6. [**Installing the more annoying Python Packages**](#installing-the-more-annoying-python-packages) - [Jetcam](#installing-jetcam), [DLib](#installing-dlib), [Tensorflow](#installing-tensorflow)
7. [**Installing Speech Recognition Packages**](#installing-speech-recognition-packages) - [Mozilla TTS](#installing-mozilla-tts), [RASA](#installing-rasa), [Deepspeech](#installing-deepspeech)

## Getting Started with the Jetson Nano
Adapted from [Nvidia's guide: Getting Started with Jetson Nano 2GB Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro).\
Prerequisites: A micro-SD card (preferably at least 32GB) and an SD card adapter for your laptop
1. Download the [Jetson Nano 2GB Developer Kit SD Card Image](https://developer.nvidia.com/jetson-nano-2gb-sd-card-image).
2. Download, install and launch [SD Memory Card Formatter for Windows](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/).
3. Format your SD card with the installed program. See the full guide for more information.
4. Download, install and launch [Etcher](https://www.balena.io/etcher).
5. Flash the downloaded SD card image to your formatted SD card with Etcher. See the full guide for more information.
6. Insert the flashed microSD into the Jetson Nano.
7. If you have a monitor, keyboard and mouse, connect them to the Nano.
8. If not, install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and connect your Nano to your laptop with a micro-USB cable. 
- Open "Device Manager", and under "Ports (COM & LPT)", find the COM port number (e.g. COM10). 
- Open PuTTY, set "Connection Type" to "Serial", input the COM port into the "Serial line" and set "Speed" to "115200", then click "Open". 
9. Connect your Nano to a power supply with a type-C cable. The supply should be 5V, so you can use a powerbank.
10. Complete the initial setup from the display or the terminal. Make sure you remember your username & password. I used "nvidia" for both. For APP partition size, use the max size suggested.
11. If running headless set-up (through micro-USB), there may be a network configuration stage. Connect the Nano to your laptop through a LAN cable and select "eth0". You might have to perform steps 2-3 in [**Configuring Nano for SSH**](#configuring-nano-for-ssh).

## Configuring Nano for SSH
If you don't have a monitor, keyboard and mouse, SSH is a good idea. To SSH, you need a USB Wifi adapter or an Ethernet cable with a USB to LAN adapter if necessary.
1. Connect the Ethernet cable between the Nano and your laptop, and connect the Nano to a power supply.
2. Click your Wifi icon in your taskbar and select "Network & Internet settings". Under "Advanced network settings", select "Change adapter options".
3. Right click the Wifi you're using and select "Properties". Under "Sharing", make sure "Allow other network users to connect through this computer's Internet connection" is checked.
4. Open Ubuntu and run the command ```ssh username@ipaddress```, e.g. ```ssh nvidia@192.168.137.215```. The IP address here should remain the same, but in case there's no connection, you can check manually by opening the Nano's terminal through PuTTY and running ```ifconfig```. The address will be under "eth0:" next to "inet". 
5. Enter the password (mine is "nvidia") and you're through. 

## Basic Setup
1. Update and upgrade your Ubuntu. The -y flag is used to confirm to our computer that we want to upgrade all the packages.
```
sudo apt-get update
sudo apt-get -y upgrade
```
### Installing Python
1. Install Python. This should install Python2.7 as python and Python3.6.9 as python3.\
```sudo apt-get install python python3```
3. Install pip for Python3.\
```sudo apt-get install -y python3-pip```
3. Install Python3.8 and its necessary packages.
```
sudo apt install python3.8
sudo apt install python3.8-venv python3.8-dev
```
5. To create a virtual environment in Python3.8, run:
```
python3.8 -m venv yourvenvname
source yourvenvname/bin/activate
```
7. Now, if you create your virtual environment and activate it, you can just use ```python``` instead of ```python3.8```. But if you're outside your environment, you have to use ```python3.8```.

### Installing others
1. Install Git.\
```sudo apt-get install git```
2. (Optional) Install your text editor of choice: nano or gedit.
```
sudo apt-get install nano
sudo apt-get install gedit
```

## Python vs Python3
When to use ```python```, ```python3``` or ```python3.8```, and especially ```pip``` vs ```pip3``` vs ```python3.8 -m pip```?
- When you run ```sudo apt-get install python python3``` and you run Python as a user (no virtual environment), running ```python``` should show the terminal for 2.7, running ```python3``` will be 3.6, and ```python3.8``` will be of course, 3.8.
- Running Python in a virtual environment will depend on which version of Python you set it up in. If you set it up with ```python3.8 -m venv myvenv```, running both ```python``` and ```python3``` will result in Python3.8.
- ```pip``` is more tricky, and will vary. The best way to find out is to run ```pip --version``` to see which it runs in. In a virtual environment, the ```pip``` and ```pip3``` will definitely be in the version of Python the environment was created, but as a user, you should check manually first. For me, both ```pip``` and ```pip3``` as a user were configured to Python3.8. 
- If ```pip``` is not tied to Python3.8, run ```python3.8 -m pip install package``` instead when you want to install as a user.

## Installing Basic Python Packages
The list of packages for the object detection and emotion recognition is: NumPy, OpenCV, DLib 19.22.1, Tensorflow 2.6, imutils, JetCam, and sk-learn.
1. Before installing anything, make sure your pip is upgraded.\
```pip install --upgrade pip```
2. Install wheel to save you some trouble.\
```pip install wheel```
3. Install the easier packages:
- ```pip install scikit-build opencv-python``` (skbuild is a module required by opencv-python)
- ```pip install imutils```
- ```pip install scikit-learn==0.24.2```
- ```pip install --upgrade numpy``` (opencv-python installs its own version of numpy, so will have to install Numpy again to avoid compile errors when installing RASA)
4. If any wheels fail to build, run step 1 again. Numpy especially tends to fail if pip is not upgraded.

## Installing the more annoying Python packages
The troublesome ones are DLib and Tensorflow, as well as JetCam for interfacing with the USB camera.

### Installing Jetcam
1. Install from source.
```
git clone https://github.com/NVIDIA-AI-IOT/jetcam
cd jetcam
sudo python3.8 setup.py install
```
2. ```cd``` out of the **jetcam/** repository and test with Python3.8:
```
> from jetcam.usb_camera import USBCamera
> cam = USBCamera(capture_device=0) # the number here can be found with "ls /dev/video*"
> cam.read()
```

**Possible error 1: ModuleNotFoundError: No module named 'jetcam.usb_camera'.**
1. Check your site-packages for a **jetcam/** folder. If running in a virtual environment, it should be ```ls yourvenv/lib/python3.8/site-packages/```, if not, it'll be ```ls /usr/local/lib/python3.8/dist-packages```. If it does not exist, go to step 2.
2. ```cd``` into the **jetcam/** folder created when you cloned from Git. If you installed into your root directory, it should be ```cd ~/jetcam```. There should be another **jetcam/** folder inside.
3. Copy that **jetcam/** folder into your site-packages folder at ```sudo cp -r ./jetcam yourvenv/lib/python3.8/site-packages/``` or ```sudo cp -r ./jetcam /usr/local/lib/python3.8/dist-packages```.
4. Test again in Python3.8.\
```> from jetcam.usb_camera import USBCamera```

**Possible error 2: ModuleNotFoundError: No module named 'traitlets'.**
1. Install the module with pip.\
```pip install traitlets```

**Possible error 3: RuntimeError: Could not read image from camera...Could not initialize camera.**
1. ```cd``` into the **jetcam/** folder you copied into your **site-packages/** or **dist-packages/** folder in error 1.
2. Open the ***usb_camera.py*** file.
3. Modify the code at line 20:
```
#self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER) ## Comment out this line
self.cap = cv2.VideoCapture(self.capture_device)
```

### Installing DLib
1. Install the dependencies:
```
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
```
2. Install DLib with pip.\
```pip install dlib==19.22.1```
4. If installation is successful, test with Python3.8.
```
> import dlib
> dlib.__version__
> dlib.get_frontal_face_detector()
```

### Installing Tensorflow
Adapted from [**Install Tensorflow 2.4 on Jetson Nano**](https://qengineering.eu/install-tensorflow-2.4.0-on-jetson-nano.html).
1. Download the dependencies
```
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev
```
2. Install Cython v0.29.26, which is required to install h5py v3.1.0.\
```pip install Cython==0.29.26```
3. Install h5py v3.1.0.\
```pip install h5py==3.1.0```
4. A whl package for Tensorflow 2.6.0 for AArch64 has been built by [KumaTea](https://github.com/KumaTea), so we'll just need to download the package and run a pip install. Huge thanks to KumaTea for building the package.
```
wget https://github.com/KumaTea/tensorflow-aarch64/releases/download/v2.6/tensorflow-2.6.0-cp38-cp38-linux_aarch64.whl
python3.8 -m pip install tensorflow-2.6.0-cp38-cp38-linux_aarch64.whl
```

## Installing Speech Recognition Packages
### Installing Mozilla TTS
1. Install dependencies
```
sudo apt-get update -y
sudo apt-get install -y pkg-config
sudo apt-get install espeak
```
2. Clone TTS repository and setup
```
git clone https://github.com/mozilla/TTS
cd TTS
git checkout 72a6ac5
sudo python3.8 setup.py develop
```
3. This is likely to terminate when Tensorflow can't be installed. Install the remaining packages manually.\
```pip install torch librosa==0.7.2 phonemizer==3.0.1 unidecode==0.4.20 inflect```
4. This version requires Numba v0.48, which requires llvmlite v0.31.0, which in turn requires LLVM 7+. We first install LLVM 7, then add the llvm-config to the path through a symbolic link. Then we install the source file for llvmlite v0.31.0, extract it and build it. After installing llvmlite, a pip install of Numba v0.48 should be successful.
```
sudo apt-get install llvm-7 
sudo ln -s /usr/bin/llvm-config-7 /usr/local/bin/llvm-config
wget https://files.pythonhosted.org/packages/17/fc/da81203725cb22d53e4f819374043bbfe3327831f3cb4388a3c020d7a497/llvmlite-0.31.0.tar.gz
tar xvf llvmlite-0.31.0.tar.gz
cd llvmlite-0.31.0
python3.8 setup.py build
pip install numba==0.48.0
```
5. PyTorch requires the updated version of Numpy. We'll have to ignore Tensorflow v2.6's warning of requiring a Numpy version ~=1.19.3.\
```pip install --upgrade numpy```

### Installing RASA
1. Install dependencies.\
```sudo apt-get install libpq-dev```
2. Install Tensorflow dependencies - tensorflow-addons v0.14.0 and tensorflow-text v2.6.0. I've built the .whl packages and uploaded under **builds/** so we'll just need to pip install it.\
```
git clone -b jetsonNano https://github.com/rachung2510/FYP70_SocialRobot.git
cd FYP70_SocialRobot/builds
pip install tensorflow_addons-0.14.0.dev0-cp38-cp38-linux_aarch64.whl
pip install tensorflow_text-2.6.0-cp38-cp38-linux_aarch64.whl
```
3. Install RASA through a downgraded version of pip.
```
pip install pip==20.2
pip install rasa==3.0.4
pip install sanic==21.9.3
```
4. Install the newest version of numpy again, if not matplotlib will raise compile errors.\
```pip install --upgrade numpy```
5. Install any additional packages which raises error with version. For me it was pyjwt being installed as v2.3.0 when RASA required v2.1.0.\
```pip install pyjwt==2.1.0```

**Possible error 1: Illegal instruction (core dumped) when running ```rasa run --enable-api```**\
This could be because getauxval did not succeed (See [original answer](https://stackoverflow.com/questions/65631801/illegal-instructioncore-dumped-error-on-jetson-nano)). Open up the .bashrc file ```nano ~/.bashrc``` and insert the line ```export OPENBLAS_CORETYPE=ARMV8``` at the bottom, then do a ```sudo reboot```.

**Possible error 2: ImportError: cannot allocate memory in static TLS block**
1. Find the very first RASA Python file which resulted in the error. For me, it was **/home/nvidia/.local/lib/python3.8/site-packages/rasa/__main__.py**.
2. Open up that file and write ```import sklearn``` at the very top (before all other import statements).

### Installing Deepspeech
TBE
