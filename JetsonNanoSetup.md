# Setting up the Jetson Nano with the necessary libraries

## Getting Started with the Jetson Nano
Adapted from [Nvidia's guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).
Prerequisites: A micro-SD card (preferably at least 64GB) and an SD card adapter for your laptop
1. Download the [Jetson Nano Developer Kit SD Card Image](https://developer.nvidia.com/jetson-nano-sd-card-image).
2. Download, install and launch [SD Memory Card Formatter for Windows](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/).
3. Format your SD card with the installed program. See the full guide for more information.
4. Download, install and launch [Etcher](https://www.balena.io/etcher).
5. Flash the downloaded SD card image to your formatted SD card with Etcher. See the full guide for more information.
6. Insert the flashed microSD into the Jetson Nano.
7a. If you have a monitor, keyboard and mouse, connect them to the Nano.
7b. If not, install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and connect your Nano to your laptop with a micro-USB cable. 
- Open "Device Manager", and under "Ports (COM & LPT)", find the COM port number (e.g. COM10). 
- Open PuTTY, set "Connection Type" to "Serial", input the COM port into the "Serial line" and set "Speed" to "115200", then click "Open". 
8. Connect your Nano to a power supply with a type-C cable. The supply should be 5V, so you can use a powerbank.
9. Complete the initial setup from the display or the terminal. Make sure you remember your username & password. I used "nvidia" for both. For APP partition size, use the max size suggested.

## Configuring Nano for SSH
If you don't have a monitor, keyboard and mouse, SSH is a good idea. To SSH, you need a USB Wifi adapter or an Ethernet cable with a USB to LAN adapter if necessary.
1. Connect the Ethernet cable between the Nano and your laptop, and connect the Nano to a power supply.
2. Click your Wifi icon in your taskbar and select "Network & Internet settings". Under "Advanced network settings", select "Change adapter options".
3. Right click the Wifi you're using and select "Properties". Under "Sharing", make sure "Allow other network users to connect through this computer's Internet connection" is checked.
4. Open Ubuntu and run the command ```ssh username@ipaddress```, e.g. ```ssh nvidia@192.168.137.100```. The IP address here should remain the same, but in case there's no connection, you can check manually by opening the Nano's terminal through PuTTY and running ```ifconfig```. The address will be under "eth0:" next to "inet". 
5. Enter the password (mine is "nvidia") and you're through. 

## Basic setup for Python
1. Run ```sudo apt-get update``` and ```sudo apt-get u-y pgrade```. The -y flag is used to confirm to our computer that we want to upgrade all the packages.
2. Install Python with ```sudo apt-get install python python3```. This should install Python2.7 as python and Python3.6.9 as python3. 
3. (Optional) Install your text editor of choice: nano or gedit with ```sudo apt-get install nano``` or ```sudo apt-get install gedit```.
4. Install pip for Python3 with sudo apt-get install -y python3-pip.
5. Install Python3.8 with ```sudo apt install python3.8```. From now on, if you want to run a program in python3.8, you must make sure your virtual environment was created in Python3.8, or you must specify ```python3.8 myprog.py```. If you want to pip install for Python3.8, run ```python3.8 -m pip install mypackage```.
6. Install the necessary packages for Python3.8 ```sudo apt install python3.8-venv python3.8-dev```.
7. To create a virtual environment in Python3.8, run ```python3.8 -m venv yourvenvname```, and activate with ```source yourvenvname/bin/activate```.
8. Now, if you create your virtual environment and activate it, you can just use ```python``` instead of ```python3.8```. But if you're outside your environment, you have to use ```python3.8```.

## Installing basic Python packages
The list of packages for the object detection and emotion recognition is: NumPy, OpenCV, DLib 19.22.1, Tensorflow 2.6, imutils, and sk-learn.
1. Before installing anything, make sure your pip is upgraded with ```python3.8 -m pip install --upgrade pip``` or just ```pip install --upgrade pip``` if you're in your virtual environment created in Python3.8.
2. Run ```pip install wheel``` to save you some trouble.
3. Install the easier packages:
- ```pip install numpy```
- ```pip install opencv-python```
- ```pip install imutils```
- ```pip install sklearn``` (this can take quite a while)

## Installing the more annoying Python packages
The troublesome ones are DLib and Tensorflow.
### Installing DLib
1. CMake is necessary to install Dlib: ```sudo apt-get install cmake```.
2. First, try ```pip install dlib``` with all the variations (```python3.8 -m pip install dlib```, ```pip3 install dlib```). If there are any successful runs, make sure you test with Python3.8 by opening ```python3.8``` then typing ```import dlib``` to see if you've successfully installed into Python3.8, or you can do the one-liner ```python3.8 -c "import dlib"``` and look out for error messages.
3. In the likely situation a pip install fails, you have to install it by source. Download the source file with ```wget https://files.pythonhosted.org/packages/f0/a2/ba6163c09fb427990180afd8d625bcecc5555af699c253193c35ffd48c4f/dlib-19.22.1.tar.gz``` into your root directory.
4. Unzip the .tar.gz file with ```tar -xvzf dlib-19.22.1.tar.gz```.
5. Enter the directory ```cd cd dlib-19.22.1/```.
6. Run ```sudo python3.8 setup.py install```. This will take a while.
7. After the above step finishes, you should have a **dlib-19.22.1-py3.8-linux-aarch64.egg** in your /usr/local/lib/python3.8/dist-packages/ directory.
8. Try running by opening ```python3.8`` and ```import dlib``` to see if it's working.

#### Possible error 1: 
*undefined symbol: png_riffle_palette_neon* when running ```import dlib```
TBE

### Installing Tensorflow
TBE
