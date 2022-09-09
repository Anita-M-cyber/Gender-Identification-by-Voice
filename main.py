# Importing section for bot
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler, CallbackQueryHandler, Filters
from pydub import AudioSegment
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

import logging
import numpy as np
import pickle

import subprocess
import os

# Importing section for audio model
import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack

# ___________________________________________________________________ #
# get token from token.conf
TOKEN = open("token.conf", "r").read().strip()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def start(bot, update):
  update.message.reply_text('Hi! Send me a vocal message and I tell you if you are "male" or "female"!')

##
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30
##

def is_silent(snd_data):
  "Returns 'True' if below the 'silent' threshold"
  return max(snd_data) < THRESHOLD


def normalize(snd_data):
  "Average the volume out"
  MAXIMUM = 16384
  times = float(MAXIMUM) / max(abs(i) for i in snd_data)

  r = array('h')
  for i in snd_data:
    r.append(int(i * times))
  return r


def trim(snd_data):
  "Trim the blank spots at the start and end"

  def _trim(snd_data):
    snd_started = False
    r = array('h')

    for i in snd_data:
      if not snd_started and abs(i) > THRESHOLD:
        snd_started = True
        r.append(i)

      elif snd_started:
        r.append(i)
    return r

  # Trim to the left
  snd_data = _trim(snd_data)

  # Trim to the right
  snd_data.reverse()
  snd_data = _trim(snd_data)
  snd_data.reverse()
  return snd_data


def add_silence(snd_data, seconds):
  "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
  r = array('h', [0 for i in range(int(seconds * RATE))])
  r.extend(snd_data)
  r.extend([0 for i in range(int(seconds * RATE))])
  return r


def extract_feature(file_name, **kwargs):
  """
  Extract feature from audio file `file_name`
      Features supported:
          - MFCC (mfcc)
          - Chroma (chroma)
          - MEL Spectrogram Frequency (mel)
          - Contrast (contrast)
          - Tonnetz (tonnetz)
      e.g:
      `features = extract_feature(path, mel=True, mfcc=True)`
  """
  mfcc = kwargs.get("mfcc")
  chroma = kwargs.get("chroma")
  mel = kwargs.get("mel")
  contrast = kwargs.get("contrast")
  tonnetz = kwargs.get("tonnetz")
  X, sample_rate = librosa.core.load(file_name)
  if chroma or contrast:
    stft = np.abs(librosa.stft(X))
  result = np.array([])
  if mfcc:
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
  if chroma:
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))
  if mel:
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
  if contrast:
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))
  if tonnetz:
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))
  return result


from utils import load_data, split_data, create_model
import argparse


def predict(bot, update):
  file_id = update.message.voice.file_id
  new_file = bot.get_file(file_id)
  new_file.download('voice.ogg')

  sound = AudioSegment.from_ogg("voice.ogg")
  sound.export("voice.wav", format="wav")

  FNULL = open(os.devnull, 'w')
  subprocess.call(("Rscript", "R/extract_feature.r"), stdout=FNULL, stderr=subprocess.STDOUT, shell=True)

  parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                  and perform inference on a sample you provide (either using your voice or a file)""")
  parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
  args = parser.parse_args()
  file = args.file
  # construct the model
  model = create_model()
  # load the saved/trained weights
  model.load_weights("results/model.h5")

  file = "voice.wav"

  # extract features and reshape it
  features = extract_feature(file, mel=True).reshape(1, -1)
  # predict the gender!
  male_prob = model.predict(features)[0][0]
  female_prob = 1 - male_prob
  gender = "male" if male_prob > female_prob else "female"
  # show the result!
  print("Result:", gender)
  print(f"Probabilities:     Male: {male_prob * 100:.2f}%    Female: {female_prob * 100:.2f}%")

  text = ""
  if gender == "male":
    # update.message.reply_text("You are male!")
    text += "you are <b>male</b>"
    text += "\n"
    #text += "probability: "
    #text += + str(int(male_prob * 100)) + "%"
  else:
    # update.message.reply_text("You are female!")
    text += "you are <b>female</b>"
    text += "\n"
    #text += "probability = "\
    #text += str(int(female_prob * 100)) + "%"




  # if gender == "male":
  #   text += "for <b>LR</b> you are <b>male</b>"
  # else:
  #   text += "for <b>LR</b> you are <b>female</b>"
  update.message.reply_text("We applied algorithms:\n"+text, parse_mode='HTML')


def main():
  updater = Updater(TOKEN)

  dp = updater.dispatcher

  dp.add_handler(MessageHandler(Filters.voice, predict))
  dp.add_handler(CommandHandler('start', start))
  dp.add_handler(CommandHandler('help', start))

  updater.start_polling()
  updater.idle()


if __name__ == '__main__':
  main()











