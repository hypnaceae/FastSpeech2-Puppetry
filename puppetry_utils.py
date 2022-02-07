# import librosa
# import numpy as np
import speech_recognition as sr
# from scipy.io import wavfile
# import sys
import os

# change this path to where you saved the charsiu package
charsiu_dir = './Charsiu'
# charsiu_dir = 'C:/Users/uwu/PycharmProjects/FastSpeechPuppetry/Charsiu'
os.chdir(charsiu_dir)

#sys.path.append('%s/src/' % charsiu_dir)

from Charsiu import Charsiu


def transcribe_audio(input_filepath):
    # get a simple ASR transcription of the audio.

    r = sr.Recognizer()
    with sr.AudioFile(input_filepath) as source:
        audio = r.record(source)  # read the entire audio file

    try:
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        transcription = r.recognize_google(audio)
        print(transcription)

        # write transcription to txt file for use by Charsiu
        with open("./puppetry-input" + 'input.txt', 'a') as transcription_file:
            transcription_file.write(transcription + '\n')
        transcription_file.close()

        return transcription

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand puppetry input audio.")
        quit()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        quit()


def align_audio(input_filepath, input_txt):
    # perform forced alignment on the audio and the transcription on the phoneme level, for duration puppetry

    # load data
    audio_file = input_filepath
    txt_file = input_txt
    csv_file = "C:/Users/uwu/PycharmProjects/FastSpeechPuppetry/puppetry-input/input.csv"

    # initialize model
    # charsiu = Charsiu.charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')

    # initialize model
    charsiu = Charsiu.charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

    # read in text file
    with open(txt_file) as f:
        text = f.read()

    print("Puppetry text detected:", text)
    # perform forced alignment
    alignment = charsiu.align(audio=audio_file, text=text)

    # perform forced alignment and save the output as a csv file
    charsiu.serve(audio=audio_file, text=text, output_format='csv', save_to=csv_file)

    # read the csv and return raw phoneme sequence as well as phoneme durations
    raw_timings = []
    raw_phonemes = []
    with open(csv_file, 'r') as csv_file:
        for line in csv_file.readlines():
            current_line = line.strip("\n").split("	")
            raw_timings.append(round(float(current_line[1]) - float(current_line[0]), 4) * 1000)
            raw_phonemes.append(current_line[2])
    csv_file.close()
    return raw_phonemes, raw_timings

def test_alignment():
    # just align input.wav and input.txt and print result
    print(align_audio("C:/Users/uwu/PycharmProjects/FastSpeechPuppetry/puppetry-input/input.wav",
                      "C:/Users/uwu/PycharmProjects/FastSpeechPuppetry/puppetry-input/input.txt"))

#test_alignment()

# how to do pitch puppetry? maybe pitch per frame? need to find out how fs2 sets pitch targets...
