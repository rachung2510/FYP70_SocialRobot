import subprocess
from pydub import AudioSegment as am

processed = 'bot_reply_processed.wav'

def process(file, mono, frame_rate):
    sound = am.from_file(file, format='mp3', frame_rate=8000)
#    sound = am.from_mp3("myfile.mp3", frame_rate=22050)
    if mono:
        sound = am.from_mono_audiosegments(sound, sound)
    sound = sound.set_frame_rate(frame_rate)
    sound.export(processed, format='wav')

def playsound(file, mono=False, frame_rate=48000, inform=True):
    if inform:
        print(" > Playing audio...")
    if file[-4:]=='.wav' and (not mono) and frame_rate==48000:
        p = subprocess.Popen(["aplay", "-D", "front:CARD=U0x19080x332a,DEV=0", file])
    else:
        process(file, mono, frame_rate)
        p = subprocess.Popen(["aplay", "-D", "front:CARD=U0x19080x332a,DEV=0", processed])
    p.wait()
    p.terminate()
