from play_audio import playsound
import argparse
from gtts import gTTS

parser = argparse.ArgumentParser()
parser.add_argument("--file", nargs="?", default="bot_reply.wav")
parser.add_argument("-s", "--stereo", action="store_true")
parser.add_argument("--tts", action="store_true")
args = parser.parse_args()

if args.tts:
    tts = gTTS("Sorry I don't understand that, let's try talking about something else")
    tts.save('msg.mp3')
    playsound('msg.mp3', mono=True)
else:
    playsound(args.file, not args.stereo)

