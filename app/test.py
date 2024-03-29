import soundfile
import playsound
from espnet2.bin.tts_inference import Text2Speech


tts = Text2Speech.from_pretrained("mio/tokiwa_midori")
print('[Info] model loaded')

tts_output = tts("こんにちは、クトリ・ノタ・ セニオリスです。 終末なにしてますか？ 忙しいですか？ 救ってもらっていいですか？")
print(tts_output)

soundfile.write("/contents/tts.wav", tts_output['wav'].numpy(), tts.fs, 'PCM_16')
playsound.playsound("/contents/tts.wav")
