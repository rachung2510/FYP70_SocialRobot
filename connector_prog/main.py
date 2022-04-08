## Run this command in terminal before executing this program
## rasa run --enable-api
## and also run this in seperate terminal
## rasa run actions

import sklearn
from Emotion_Recognition import init_emotion, get_emotion_class
from Object_Detection import SimonSays_item
from Nothing_Detection import SimonSays_nothing
import Scissor_Paper_Stone as SPS
import PopTheBubble as PTB
import ShowMeTheNumber as SMTN
#from stt_tts import load_tts, tts
import speech_recognition as sr
from play_audio import playsound
from threading import Thread, Event
import requests
import subprocess
#from jetcam.usb_camera import USBCamera
from VideoCapture import VideoCapture
from gtts import gTTS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mic", type=int, nargs="?", default=14)
parser.add_argument("--cam", type=int, nargs="?", default=0)
args = parser.parse_args()

# =============================================================================
# Functions
# =============================================================================
def get_input():
    global message, finished
    if not finished.isSet():
        '''STT block'''
        #using speech_recognition library
        r = sr.Recognizer()  # initialize recognizer
        with sr.Microphone(device_index=args.mic) as source:
            r.adjust_for_ambient_noise(source)
            print("Speak Anything :")
            playsound("siri_start.wav", inform=False)
            audio = r.listen(source)  # listen to the source
            try:
                message = r.recognize_google(audio)  # use google recognizer to convert our audio into text
                print("I say: "+ message)
                finished.set()
            except:
                print("Sorry could not recognize your voice")  # In case of voice not recognized clearly
    return

def pred_emotion(vs, detector, predictor, models):
    global message, finished
    c = 0
    pred = "neutral"
    while not finished.isSet():
        frame = vs.read()
        emotion_class = get_emotion_class(frame, detector, predictor, models)
        if emotion_class != "neutral":
            c += 1 if emotion_class == pred else 0
            pred = emotion_class
            if c==5:
                message = emotion_class
                print("Detected emotion: " + emotion_class)
                finished.set()
    return emotion_class

# =============================================================================
# Initialisation
# =============================================================================

print("[INFO] Starting initialisation...")
emo_mode = True
game_mode = "none"
prev_game_mode = game_mode # whether it was previously game mode
SimonsaysAns = "none"
finished = Event() # input thread condition
bye_list = ["Bye bye", "Goodbye", "See you again!", "Let's talk again next time!"]

# Load camera
print("[INFO] camera sensor warming up...")
vs = VideoCapture(args.cam, width=640, height=480)
#vs = USBCamera(capture_device=args.cam, width=820, height=480)

# Load emotion model
detector, predictor, models = init_emotion()

# =============================================================================
# Main Loop
# =============================================================================

while True:

    ''' Input and output '''
    # Reset messages
    message = ""
    bot_message = ""

    # Get rasa slots for choosing input
    url = 'http://localhost:5005/conversations/default/tracker?include_events=NONE'
    s = requests.request("GET", url, headers={}, data={})
    j = s.json()

    emo_mode = j['slots']['emo_mode'] # Emotion detection slot
    game_mode = j['slots']['game_mode']  #Game mode slot
    SimonsaysAns = j['slots']['SimonsaysAns'] #Simon says slot
    SPSmessage = j['slots']['SPSmessage'] # Scissors paper stone slot
    PTBstatus = j['slots']['PTBstatus'] # Pop the bubble slot
    SMTNstatus = j['slots']['SMTNstatus'] # Show me the number slot
#    print(f"Emotion Detection mode: {emo_mode}")
#    print(f"Game mode: {game_mode}")
#    print(f"SimonsaysAns: {SimonsaysAns}")
#    print(f"SPSmessage: {SPSmessage}")
#    print(f"PTBstatus: {PTBstatus}")
#    print(f"SMTNstatus: {SMTNstatus}")

    # Choosing input message
    if emo_mode and game_mode == "none": # emotion detection and STT
        # print("Reading Emotion and STT")
#        p = subprocess.Popen(['python3.8', 'display_img.py']) # display listening img
        while not finished.isSet():
            worker = Thread(target=get_input) # STT thread
            worker.setDaemon(True)
            worker.start()
            emotion_class = pred_emotion(vs, detector, predictor, models)
            playsound("siri_stop.wav", inform=False)
#        p.kill()
    elif SimonsaysAns != "none": # if Simon says object detected
        # print("Sending Simon says security code")
        message = "dfgdyttvyhtf1559716hkyk"
    elif SPSmessage != "none": # if scissors paper stone object detected
        # print("Sending scissors paper stone security code")
        message = "BVHGGTHY4665fger45225"
    elif PTBstatus != "none": # if pop the bubble object detected
        # print("Sending pop the bubble security code")
        message = "sagrnyfvyutccreqwbtrth0t658dfb0"
    elif SMTNstatus != "none": # if show me the number object detected
        # print("Sending show me the number security code")
        message = "hgjtytn5t2GHEBLLOUIEKVF6565"
    else: # STT only
        # print("Reading STT only")
#        p = subprocess.Popen(['python3.8', 'display_img.py']) # display listening img
        get_input()
        playsound("siri_stop.wav", inform=False)
#        p.kill()

    # If there is a message, pass to rasa and print response
    if message != "":
        r = requests.post('http://localhost:5005/webhooks/rest/webhook', json={"message": message})
        for i in r.json():
            bot_message = i['text']
            print(f"{bot_message}")

    # If there is a response, pass to TTS output
    if bot_message != "":
        sentence = bot_message
#        align, spec, stop_tokens, wavform = tts(model, vocoder_model, speaker_id, sentence, TTS_CONFIG, use_cuda, ap, OUT_FILE, use_gl=False)
        tts = gTTS(sentence, tld='ca')
        tts.save("bot_reply.mp3")
        playsound("bot_reply.mp3", mono=True, frame_rate=43000) # Playing the converted file

    # End program if bot said goodbye
    if bot_message in bye_list:
        break


    ''' Object Detection '''
    # Get object detection slots to determine what to detect
    url = 'http://localhost:5005/conversations/default/tracker?include_events=NONE'
    s = requests.request("GET", url, headers={}, data={})
    j = s.json()

    object_detection = j['slots']['object_detection'] # for Simon says
    SPSflag = j['slots']['SPSflag'] # for scissors paper stone
    PTBflag = j['slots']['PTBflag'] # for pop the bubble
    SMTNflag = j['slots']['SMTNflag'] # for show me the number
    # print("object_detection: ", object_detection)
    # print("flags: ", SPSflag, PTBflag, SMTNflag)

    # Run object detection
    set_url = 'http://localhost:5005/conversations/default/tracker/events?include_events=NONE'
    if object_detection == "yes":
        item = j['slots']['item']
        ans = SimonSays_item(item,vs)
        #print(f"detected {ans}")
        r = requests.post(set_url, json={"event":"slot","name":"SimonsaysAns","value":ans, "timestamp":0})
    elif object_detection == "no":
        item = j['slots']['item']
        ans = SimonSays_nothing(item,vs)
        #print(f"detected {ans}")
        r = requests.post(set_url, json={"event":"slot","name":"SimonsaysAns","value":ans, "timestamp":0})
    elif SPSflag == True:
        SPSmessage = SPS.scissorPaperStone(vs)
        r = requests.post(set_url, json={"event":"slot","name":"SPSmessage","value":SPSmessage, "timestamp":0})
    elif PTBflag == True:
        status, duration = PTB.PopTheBubble(vs)
        r = requests.post(set_url, json={"event":"slot","name":"PTBstatus","value":str(status), "timestamp":0})
        r = requests.post(set_url, json={"event":"slot","name":"PTBduration","value":duration, "timestamp":0})
    elif SMTNflag == True:
        status, duration = SMTN.showMeTheNumber(vs)
        r = requests.post(set_url, json={"event":"slot","name":"SMTNstatus","value":str(status), "timestamp":0})
        r = requests.post(set_url, json={"event":"slot","name":"SMTNduration","value":duration, "timestamp":0})

    finished = Event() # reset input thread event
