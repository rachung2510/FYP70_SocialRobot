# Setting up the Jetson Nano with the necessary libraries

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
4. Open Ubuntu and run the command ```ssh username@ipaddress```, e.g. ```ssh nvidia@192.168.137.100```. The IP address here should remain the same, but in case there's no connection, you can check manually by opening the Nano's terminal through PuTTY and running ```ifconfig```. The address will be under "eth0:" next to "inet". 
5. Enter the password (mine is "nvidia") and you're through. 

## Basic Setup
1. Run ```sudo apt-get update``` and ```sudo apt-get -y upgrade```. The -y flag is used to confirm to our computer that we want to upgrade all the packages.
### Installing Python
1. Install Python with ```sudo apt-get install python python3```. This should install Python2.7 as python and Python3.6.9 as python3.
2. Install pip for Python3 with ```sudo apt-get install -y python3-pip```.
3. Install Python3.8 with ```sudo apt install python3.8```. From now on, if you want to run a program in python3.8, you must make sure your virtual environment was created in Python3.8, or you must specify ```python3.8 myprog.py```. If you want to pip install for Python3.8, run ```python3.8 -m pip install mypackage```.
4. Install the necessary packages for Python3.8 ```sudo apt install python3.8-venv python3.8-dev```.
5. To create a virtual environment in Python3.8, run ```python3.8 -m venv yourvenvname```, and activate with ```source yourvenvname/bin/activate```.
6. Now, if you create your virtual environment and activate it, you can just use ```python``` instead of ```python3.8```. But if you're outside your environment, you have to use ```python3.8```.

### Installing others
1. Install Git with ```sudo apt-get install git```.
2. (Optional) Install your text editor of choice: nano or gedit with ```sudo apt-get install nano``` or ```sudo apt-get install gedit```.

## Python vs Python3
When to use ```python```, ```python3``` or ```python3.8```, and especially ```pip``` vs ```pip3``` vs ```python3.8 -m pip```?
- When you run ```sudo apt-get install python python3``` and you run Python as a user (no virtual environment), running ```python``` should show the terminal for 2.7, running ```python3``` will be 3.6, and ```python3.8``` will be of course, 3.8.
- Running Python in a virtual environment will depend on which version of Python you set it up in. If you set it up with ```python3.8 -m venv myvenv```, running both ```python``` and ```python3``` will result in Python3.8.
- ```pip``` is more tricky, and will vary. The best way to find out is to run ```pip --version``` to see which it runs in. In a virtual environment, the ```pip``` and ```pip3``` will definitely be in the version of Python the environment was created, but as a user, you should check manually first. For me, both ```pip``` and ```pip3``` as a user were configured to Python3.8. 
- If ```pip``` is not tied to Python3.8, run ```python3.8 -m pip install package``` instead when you want to install as a user.

## Installing Basic Python Packages
The list of packages for the object detection and emotion recognition is: NumPy, OpenCV, DLib 19.22.1, Tensorflow 2.6, imutils, JetCam, and sk-learn.
1. Before installing anything, make sure your pip is upgraded with ```python3.8 -m pip install --upgrade pip``` or just ```pip install --upgrade pip``` if you're in your virtual environment created in Python3.8.
2. Run ```pip install wheel``` to save you some trouble.
3. Install the easier packages:
- ```pip install numpy```
- ```pip install scikit-build opencv-python``` (skbuild is a module required by opencv-python)
- ```pip install imutils```
- ```pip install sklearn``` (this can take quite a while)
4. If any wheels fail to build, run step 1 again. Numpy especially tends to fail if pip is not upgraded.

## Installing the more annoying Python packages
The troublesome ones are DLib and Tensorflow, as well as JetCam for interfacing with the USB camera.

### Installing Jetcam
1. ```git clone https://github.com/NVIDIA-AI-IOT/jetcam```.
2. ```cd jetcam```.
3. ```sudo python3.8 setup.py install```.
4. To test, ```cd``` into the root directory or activate your virtual environment and run ```python3.8```. Then:
```
> from jetcam.usb_camera import USBCamera
> cam = USBCamera(capture_device=0) # the number here can be found with "ls /dev/video*"
> cam.read()
```

#### Possible error 1:
ModuleNotFoundError: No module named 'jetcam.usb_camera'.
In this case, you're probably running in a virtual environment.
1. Check your site-packages at ```ls yourvenv/lib/python3.8/site-packages/``` for a **jetcam/** folder. If it does not exist, go to step 2.
2. ```cd``` into the **jetcam/** folder created when you cloned from Git. If you installed into your root directory, it should be ```cd ~/jetcam```. There should be another **jetcam/** folder inside.
3. Copy that **jetcam/** folder into **yourvenv/lib/python3.8/site-packages/**.
4. Try again with ```python3.8``` and ```from jetcam.usb_camera import USBCamera```.

#### Possible error 2:
ModuleNotFoundError: No module named 'traitlets'.
1. ```pip install traitlets``` if in a virtual environment or ```python3.8 -m pip install traitlets``` if otherwise.

#### Possible error 3:
RuntimeError: Could not read image from camera.
1. ```cd``` into the **jetcam/** folder created when you cloned from Git, and then ```cd``` into the **jetcam/** folder inside.
2. Open the ***usb_camera.py*** file.
3. Modify the code at line 20:
#self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER) ## Comment out this line
self.cap = cv2.VideoCapture(self.capture_device)
4. If you encountered error 1 and copied the **jetcam/** folder into your virtual environment site-packages, make the necessary changes on there as well.

### Installing DLib
1. CMake is necessary to install Dlib: ```sudo apt-get install cmake```.
2. First, try ```pip install dlib``` with all the variations (```python3.8 -m pip install dlib```, ```pip3 install dlib```). If there are any successful runs, make sure you test with Python3.8 by opening ```python3.8``` then typing ```import dlib``` to see if you've successfully installed into Python3.8, or you can do the one-liner ```python3.8 -c "import dlib"``` and look out for error messages.
3. In the likely situation a pip install fails, you have to install it by source. 
4. Do the following steps:
```
git clone https://github.com/davisking/dlib
cd dlib
mkdir build
cd build/
cmake ..
cmake --build .
cd ..
sudo python3.8 setup.py install
```
5. Try running by opening ```python3.8``` and ```import dlib``` then ```dlib.__version__``` or ```dlib.get_frontal_face_detector()``` to see if it's working.

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
```python3.8 -m pip install Cython==0.29.26```
3. Install h5py v3.1.0.\
```python3.8 -m pip install h5py==3.1.0```
4. Download the whl file from [**KumaTea/tensorflow-aarch64**](https://github.com/KumaTea/tensorflow-aarch64/releases?q=2.6&expanded=true). The link below is for Python 3.8 and Tensorflow 2.6. For installation without virtual environment (recommended since importing tensorflow in a virtual environment tends to result in "core dumped" error), download the whl file without "manylinux".\
```wget https://github.com/KumaTea/tensorflow-aarch64/releases/download/v2.6/tensorflow-2.6.0-cp38-cp38-linux_aarch64.whl```
5. Run a pip install on the whl file.\
```python3.8 -m pip install tensorflow-2.6.0-cp38-cp38-linux_aarch64.whl```
