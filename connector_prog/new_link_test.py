## Run this command in terminal before executing this program
## rasa run --enable-api
## and also run this in seperate terminal
## rasa run actions

import requests
from emotion_recognition import init_emotion, get_emotion_class
from stt_tts import load_tts, tts, record_audio
from threading import Thread, Event
from play_audio import playsound
from deepspeech import Model
import scipy.io.wavfile as wav
import os

'''Settings for STT'''
WAVE_OUTPUT_FILENAME = "user_audio.wav"
model_file_path = 'stt_tts_data/deepspeech-0.9.3-models.pbmm'

def get_input():
    global message, finished
    if not finished.isSet():
        #message = input("Your input: \n")
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


'''Connector Program'''

bot_message = ""
message=""
emo_mode = True
finished = Event()

bye_list = ["Bye", "Goodbye", "See you again!", "Let's talk again next time!"]

#Load TTS
model, vocoder_model, speaker_id, TTS_CONFIG, use_cuda, ap, OUT_FILE = load_tts()

#Load emotion model
vs, detector, predictor, models = init_emotion()

# Main prog loop
while True:

    if emo_mode:
        while not finished.isSet():
            worker = Thread(target=get_input) # STT thread
            worker.setDaemon(True)
            worker.start()
            emotion_class = pred_emotion(vs, detector, predictor, models)
    else:
        get_input()

    #if len(message)==0:
    #    continue
    print("Sending message now...")

    #Pass message to rasa and print response
    r = requests.post('http://localhost:5005/webhooks/rest/webhook', json={"message": message})
    for i in r.json():
        bot_message = i['text']
        print(f"{bot_message}")
    
    #Get rasa slot for emotion detection mode
    url = "http://localhost:5005/conversations/default/tracker?include_events=NONE"
    s = requests.request("GET", url, headers={}, data={})
    j = s.json()
    emo_mode = j['slots']['emo_mode']
    print(f"Emotion Detection mode: {emo_mode}")

    # TTS block
    if bot_message != "":
        sentence = bot_message
        align, spec, stop_tokens, wavform = tts(model, vocoder_model, speaker_id, sentence, TTS_CONFIG, use_cuda, ap, OUT_FILE, use_gl=False)
        playsound("bot_reply.wav") # Playing the converted file
        
    finished = Event() # reset event
    
    # End program if bot said goodbye
    if bot_message in bye_list:
        break
