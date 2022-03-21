import subprocess
from pydub import AudioSegment as am

processed = 'bot_reply_processed.wav'

def process(file, mono):
    sound = am.from_file(file, format='mp3', frame_rate=8000)
#    sound = am.from_mp3("myfile.mp3", frame_rate=22050)
    if mono:
        sound = am.from_mono_audiosegments(sound, sound)
    sound = sound.set_frame_rate(44000)
    sound.export(processed, format='wav')

def playsound(file, mono=True):
    print(" > Playing audio...")
    process(file, mono)
    p = subprocess.Popen(["aplay", "-D", "front:CARD=U0x19080x332a,DEV=0", processed])
    p.wait()
    p.terminate()
