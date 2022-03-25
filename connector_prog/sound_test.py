from play_audio import playsound
import argparse
from gtts import gTTS

default = "bot_reply.mp3"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", nargs="?", default=default)
parser.add_argument("-tts", "--tts", action="store_true")
parser.add_argument("-m", "--mono", action="store_true")
parser.add_argument("-fr", "--frame_rate", type=int, default=48000)
args = parser.parse_args()

if args.tts:
#    tts = gTTS("Sorry I don't understand. Could you please repeat yourself?", lang="zh-CN")
    tts = gTTS("Sorry I don't understand. Could you please repeat yourself?", tld='ca')
    tts.save('msg.mp3')
    playsound('msg.mp3', mono=True, frame_rate=44000)
else:
    if args.file == default:
        playsound(args.file, mono=True, frame_rate=42000)
    else:
        playsound(args.file, args.mono, args.frame_rate)

