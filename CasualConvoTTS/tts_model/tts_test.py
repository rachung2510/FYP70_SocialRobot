import os
import torch
import time
#import IPython

from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator

def tts(model, text, CONFIG, use_cuda, ap, OUT_FILE, use_gl):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None, truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
    # mel_postnet_spec = ap.denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
        waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    #IPython.display.display(IPython.display.Audio(waveform, rate=CONFIG.audio['sample_rate']))
    ap.save_wav(waveform, OUT_FILE)  
    return alignment, mel_postnet_spec, stop_tokens, waveform

# runtime settings
use_cuda = False

# model paths
TTS_MODEL = "data/tts_model.pth.tar"
TTS_CONFIG = "data/config.json"
VOCODER_MODEL = "data/vocoder_model.pth.tar"
VOCODER_CONFIG = "data/config_vocoder.json"
OUT_FILE = "tts_out.wav"

# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# load the audio processor
TTS_CONFIG.audio['stats_path'] = 'data/scale_stats.npy'
ap = AudioProcessor(**TTS_CONFIG.audio)   

# LOAD TTS MODEL
# multi speaker 
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load model state
cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])


# LOAD VOCODER MODEL
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0

ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])    
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()

sentence =  "Bill got in the habit of asking himself ???Is that thought true???? and if he wasn???t absolutely certain it was, he just let it go."
align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, OUT_FILE, use_gl=False)
