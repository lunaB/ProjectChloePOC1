from espnet2.bin.tts_inference import Text2Speech
import soundfile
import playsound

tts = Text2Speech.from_pretrained("imdanboy/jets")
tts_output = tts("Hello my name is jin.")
soundfile.write("contents/tts.wav", tts_output['wav'].numpy(), tts.fs, 'PCM_16')
playsound.playsound("contents/tts.wav")
