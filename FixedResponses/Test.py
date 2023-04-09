import pyaudio
from Object_Detection import SimonSays_item
from Nothing_Detection import SimonSays_nothing
from deepspeech import Model
import requests
import scipy.io.wavfile as wav
import wave
import os
import time
import subprocess
import Scissor_Paper_Stone as SPS
import PopTheBubble as PTB
import ShowMeTheNumber as SMTN

# WAVE_OUTPUT_FILENAME = "test_audio.wav"
# model_file_path = 'deepspeech/deepspeech-0.9.3-models.pbmm'

# def record_audio(WAVE_OUTPUT_FILENAME):
# 	CHUNK = 1024
# 	FORMAT = pyaudio.paInt16
# 	CHANNELS = 1
# 	RATE = 16000
# 	RECORD_SECONDS = 8

# 	p = pyaudio.PyAudio()

# 	stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)
# 	print("Recording")
# 	frames = [stream.read(CHUNK) for i in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
# 	print("Done recording")
# 	stream.stop_stream()
# 	stream.close()
# 	p.terminate()

# 	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# 	wf.setnchannels(CHANNELS)
# 	wf.setsampwidth(p.get_sample_size(FORMAT))
# 	wf.setframerate(RATE)
# 	wf.writeframes(b''.join(frames))
# 	wf.close()

bot_message = ""
while bot_message != "Bye" or bot_message!='thanks':
# 	# read user input
# 	record_audio(WAVE_OUTPUT_FILENAME)
	
# 	# convert audio to text
# 	N_FEATURES = 25
# 	N_CONTEXT = 9
# 	BEAM_WIDTH = 500
# 	LM_ALPHA = 0.75
# 	LM_BETA = 1.85
# 	ds = Model(model_file_path)
# 	fs, audio = wav.read(WAVE_OUTPUT_FILENAME)
# 	message = ds.stt(audio)
	
# 	# remove audio file
# 	os.remove(WAVE_OUTPUT_FILENAME)
	# Get item slot
	url = "http://localhost:5005/conversations/default/tracker?include_events=NONE"
	s = requests.request("GET", url, headers={}, data={})
	j = s.json()
	SimonsaysAns = j['slots']['SimonsaysAns']
	s = requests.request("GET", url, headers={}, data={})
	j = s.json()
	SPSmessage = j['slots']['SPSmessage']
	s = requests.request("GET", url, headers={}, data={})
	j = s.json()
	PTBstatus = j['slots']['PTBstatus']
	s = requests.request("GET", url, headers={}, data={})
	j = s.json()
	SMTNstatus = j['slots']['SMTNstatus']
	if SPSmessage != "none":
		message = "BVHGGTHY4665fger45225"
	elif PTBstatus != "none":
		message = "sagrnyfvyutccreqwbtrth0t658dfb0"
	elif SMTNstatus != "none":
		message = "hgjtytn5t2GHEBLLOUIEKVF6565"
	elif SimonsaysAns != "none":
		message = "dfgdyttvyhtf1559716hkyk"
	else:
		message = input("Input: ")
		# repeat if no input
		if len(message)==0:
			continue
		
	
		
	# connect to RASA server
	r = requests.post('http://localhost:5005/webhooks/rest/webhook', json={"message": message})



	# for reference
	# print("I say: "+ message)

	print("Bot says, ")
	for i in r.json():
		bot_message = i['text']
		print(f"{bot_message}")

	s = requests.request("GET", url, headers={}, data={})
	j = s.json()
	object_detection = j['slots']['object_detection']
	if object_detection == "yes":
		item = j['slots']['item']
		ans = SimonSays_item(item)
		if ans:
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SimonsaysAns","value":True, "timestamp":0})
		else:
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SimonsaysAns","value":False, "timestamp":0})
	elif object_detection == "no":
		item = j['slots']['item']
		ans = SimonSays_nothing(item)
		if ans:
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SimonsaysAns","value":True, "timestamp":0})
		else:
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SimonsaysAns","value":False, "timestamp":0})
	else:
		s = requests.request("GET", url, headers={}, data={})
		j = s.json()
		SPSflag = j['slots']['SPSflag']
		s = requests.request("GET", url, headers={}, data={})
		j = s.json()
		PTBflag = j['slots']['PTBflag']
		s = requests.request("GET", url, headers={}, data={})
		j = s.json()
		SMTNflag = j['slots']['SMTNflag']
		print("flags: ", SPSflag, PTBflag, SMTNflag)
		if SPSflag == True:
			SPSmessage = SPS.scissorPaperStone()
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SPSmessage","value":SPSmessage, "timestamp":0})
		elif PTBflag == True:
			status, duration = PTB.PopTheBubble()
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"PTBstatus","value":str(status), "timestamp":0})
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"PTBduration","value":duration, "timestamp":0})
		elif SMTNflag == True:
			status, duration = SMTN.showMeTheNumber()
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SMTNstatus","value":str(status), "timestamp":0})
			r = requests.post('http://localhost:5005/conversations/default/tracker/events?include_events=NONE', json={"event":"slot","name":"SMTNduration","value":duration, "timestamp":0})