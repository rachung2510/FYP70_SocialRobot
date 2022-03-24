# EDU-Bot Installation on Windows with conda

This is a guide on how to install the necessary packages and run EDU-Bot on Windows with conda.

## Environment
Create a virtual environment with Anaconda for Python 3.8.
```bash
conda create -n venv python=3.8
conda activate venv
```
Clone this repository and switch branch.
```bash
git clone https://github.com/rachung2510/FYP70_SocialRobot
git checkout EDU-Bot
```

## Install Rasa
1. Install [Visual Studio Redistributable](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170). Choose x64 for 64-bit system.
2. Install some dependencies
```bash
conda install ujson
conda install tensorflow
```
3. Install Rasa with downgraded version of pip
```bash
pip install pip==20.2
pip install rasa==3.0.4
```
4. Downgrade sanic
```bash
pip install sanic==21.9.3
```

## Install STT
1. Install _speechrecognition_ and _pyaudio_
```bash
pip install SpeechRecognition PyAudio
```
If you want to use _deepspeech_ instead, you can install it
```bash
pip install deepspeech
```
2. Download the deepspeech .pbmm model from [here](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3) and
put in FYP70_SocialRobot/EDU-Bot/connector_prog/stt_tts_data.

## Install TTS
1. Download [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) for Windows. Choose x64.msi for 64-bit system.
2. Go to the control panel and edit environment variables. Add a system variable PHONEMIZER_ESPEAK_LIBRARY with the value
C:\...\eSpeak NG\libespeak-ng.dll. For example if espeak-ng is installed in C:\Program Files, then the value is 
C:\Program Files\eSpeak NG\libespeak-ng.dll.
3. Add a conda environment variable to prevent a unicode error.
```bash
conda env config vars set PYTHONUTF8=1
```
4. Install git if necessary, clone TTS repository, git checkout to a previous commit as code uses old commit
```bash
conda install git
git clone https://github.com/mozilla/TTS
cd TTS
git checkout 72a6ac5
```
5. Install TTS using setup.py
```bash
python setup.py develop
```
6. Install libsndfile
```bash
conda install -c conda-forge libsndfile
```
7. Download the mozilla [TTS](https://drive.google.com/file/d/1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos/view?usp=sharing) and 
[vocoder](https://drive.google.com/file/d/1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K/view?usp=sharing) models;
name them as tts_model.pth.tar and vocoder_model.pth.tar respectively, then
place them in FYP70_SocialRobot/EDU-Bot/connector_prog/stt_tts_data.

## Install libraries for emotion recognition
1. Install Pytorch. For CPU only:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
For GPU support with CUDA:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
2. Install libraries
```bash
conda install -c anaconda numpy
conda install -c anaconda scikit-learn
conda install -c conda-forge imutils
conda install -c conda-forge opencv
conda install -c anaconda cmake
conda install -c conda-forge dlib
conda install -c conda-forge pickle5
```

## Install libraries for Object Detection games
For Finger Detection and Plotting of Hand Land Marks:
```bash
pip install mediapipe
pip install matplotlib
```
Cutomised Object Detection:
Weights: [iteration3_final.weights](https://drive.google.com/file/d/1BQF_CJWqCWHoAAl4iFkeKmAzQBwGDYXy/view?usp=sharing) and put in FYP70_SocialRobot/EDU-Bot/connector_prog/object_data.

## Run EDU-Bot
Open 3 terminals. In terminal 1:
```bash
cd FYP70_SocialRobot/EDU-Bot
rasa run actions
```
In terminal 2:
```bash
cd FYP70_SocialRobot/EDU-Bot
rasa run --enable-api
```
In terminal 3:
```bash
cd FYP70_SocialRobot/EDU-Bot/connector_prog
python main.py
```

## Customise main file
To use _deepspeech_ instead of _speechrecognition_ for STT, comment/uncomment parts in _get_input()_.
To disable cartoon image pop-up during listening, comment lines containing _subprocess_ and _p.kill()_.
