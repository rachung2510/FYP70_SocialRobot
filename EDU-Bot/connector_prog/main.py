## Run this command in terminal before executing this program
## rasa run --enable-api
## and also run this in seperate terminal
## rasa run actions

from emotion_recognition import init_emotion, get_emotion_class
from Object_Detection import SimonSays_item
from Nothing_Detection import SimonSays_nothing
from stt_tts import load_tts, tts, record_audio
from deepspeech import Model
from play_audio import playsound
from threading import Thread, Event
import requests
import scipy.io.wavfile as wav
import os
import time
from imutils.video import VideoStream

''' Settings for STT '''
WAVE_OUTPUT_FILENAME = "user_audio.wav"
model_file_path = 'stt_tts_data/deepspeech-0.9.3-models.pbmm'

''' Define functions '''
def get_input():
    global message, finished
    if not finished.isSet():
        '''STT block'''
       	# read user input
        record_audio(WAVE_OUTPUT_FILENAME)
       	
       	# convert audio to text
        N_FEATURES = 25
        N_CONTEXT = 9
        BEAM_WIDTH = 500
        LM_ALPHA = 0.75
        LM_BETA = 1.85
        ds = Model(model_file_path)
        fs, audio = wav.read(WAVE_OUTPUT_FILENAME)
        message = ds.stt(audio)
       	
        # remove audio file
        os.remove(WAVE_OUTPUT_FILENAME)
        
       	# for reference
        print("I say: "+ message)
        finished.set()
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
                print("I'm " + emotion_class)
                finished.set()
    return emotion_class


''' Initialise '''
emo_mode = True
game_mode = "none"
prev_game_mode = game_mode # whether it was previously game mode
SimonsaysAns = "none"
finished = Event() # input thread condition
bye_list = ["Bye", "Goodbye", "See you again!", "Let's talk again next time!"]

# Load TTS
model, vocoder_model, speaker_id, TTS_CONFIG, use_cuda, ap, OUT_FILE = load_tts()

# Load emotion model
vs, detector, predictor, models = init_emotion()

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
    
    emo_mode = j['slots']['emo_mode'] #Emotion detection slot
    # print(f"Emotion Detection mode: {emo_mode}")
    
    SimonsaysAns = j['slots']['SimonsaysAns'] #Simon Says slot
    # print(f"SimonsaysAns: {SimonsaysAns}")

    # Choosing input message
    if emo_mode and game_mode == "none": # emotion detection and STT
        print("Reading Emotion and STT")
        # message = input("Your input: \n")
        while not finished.isSet():
            worker = Thread(target=get_input) # STT thread
            worker.setDaemon(True)
            worker.start()
            emotion_class = pred_emotion(vs, detector, predictor, models)
    elif SimonsaysAns != "none": # if game object detected
        print("Sending Simon says security code")
        message = "dfgdyttvyhtf1559716hkyk"
    else: # STT only
        print("Reading STT only")
        get_input()
        # message = input("Your input: \n")

    #Pass message to rasa and print response
    r = requests.post('http://localhost:5005/webhooks/rest/webhook', json={"message": message})
    for i in r.json():
        bot_message = i['text']
        print(f"{bot_message}")

    # TTS output
    if bot_message != "":
        sentence = bot_message
        align, spec, stop_tokens, wavform = tts(model, vocoder_model, speaker_id, sentence, TTS_CONFIG, use_cuda, ap, OUT_FILE, use_gl=False)
        playsound("bot_reply.wav") # Playing the converted file

    # End program if bot said goodbye
    if bot_message in bye_list:
        break


    ''' Camera Switching and Object Detection '''

    #Get game mode slot for camera switching
    url = "http://localhost:5005/conversations/default/tracker?include_events=NONE"
    s = requests.request("GET", url, headers={}, data={})
    j = s.json()
    
    game_mode = j['slots']['game_mode']  
    # print(f"Game mode: {game_mode}")
    
    # switch camera for appropriate purpose
    if prev_game_mode != game_mode:
        if game_mode == "none":
            # If mode changed from game to chat, restart emotion video stream
            print("[INFO] camera sensor warming up...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
            print("done")
        else:
            # If mode changed from chat to game, terminate emotion video stream
            # so the object detection can use it
            vs.stop()

    prev_game_mode = game_mode # update previous mode
    
    # Get object detection slot to determine whether to detect
    object_detection = j['slots']['object_detection']
    # print("object_detection: ", object_detection)
    
    # Run object detection
    set_url = 'http://localhost:5005/conversations/default/tracker/events?include_events=NONE'
    if object_detection == "yes":
        item = j['slots']['item']
        ans = SimonSays_item(item)
        #print(f"detected {ans}")
        r = requests.post(set_url, json={"event":"slot","name":"SimonsaysAns","value":ans, "timestamp":0})
    elif object_detection == "no":
        item = j['slots']['item']
        ans = SimonSays_nothing(item)
        #print(f"detected {ans}")
        r = requests.post(set_url, json={"event":"slot","name":"SimonsaysAns","value":ans, "timestamp":0})


    finished = Event() # reset input thread event
