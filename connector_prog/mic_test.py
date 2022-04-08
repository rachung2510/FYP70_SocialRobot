import speech_recognition as sr
import argparse
import cv2
import pyaudio

parser = argparse.ArgumentParser()
parser.add_argument("--dev", type=int, default=14)
parser.add_argument("--list", action='store_true')
args = parser.parse_args()

def stt(r):
    while True:
        with sr.Microphone(device_index=args.dev) as source:
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("Say something :")
            audio = r.listen(source)  # listen to the source
            try:
                message = r.recognize_google(audio)  # use google recognizer to convert our audio into text
                print("I say: "+ message)
            except:
                print("Sorry could not recognize your voice")  # In case of voice not recognized clearly

if args.list:
    lst = sr.Microphone.list_microphone_names()
    for i in range(len(lst)):
        print(i, lst[i])
else:
    r = sr.Recognizer()  # initialize recognizer
    while True:
        stt(r)
