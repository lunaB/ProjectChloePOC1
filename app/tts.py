from espnet2.bin.tts_inference import Text2Speech
import soundfile

class TTS:
  def __init__(self, model="imdanboy/kss_jets"):
    """
    Text-to-Speech class.
    Args:
      model (str): The voice model to use. e.g. 'mio/tokiwa_midori', 'imdanboy/kss_jets'
    """
    self.tts_model = Text2Speech.from_pretrained(model)

  def text_to_speech(self, text, output_file):
    tts_output = self.tts_model(text)
    soundfile.write(output_file, tts_output['wav'].numpy(), self.tts_model.fs, 'PCM_16')

# from app.logger import get_logger, wait_log
# logger = get_logger(__name__)

# # # disable warnings
# # warnings.filterwarnings("ignore")
# # logging.disable(logging.WARNING)
# # logger.setLevel(logging.INFO)
# # logger.info("Logger initialized")

# from dotenv import load_dotenv
# load_dotenv()

# # Model loading
# tts_model = None
# with wait_log(logger, "TTS model loading..."):
#   tts_model = Text2Speech.from_pretrained("mio/tokiwa_midori")


# # Function for text-to-speech
# def text_to_speech(text, output_file):
#   tts_output = tts_model(text)
#   soundfile.write("contents/tts.wav", tts_output['wav'].numpy(), tts_model.fs, 'PCM_16')
#   playsound.playsound("contents/tts.wav")


# test code
if __name__ == "__main__":
  text = '''
  중 하나로 갈 수 있게 하는 레버를 당기는 상황을 말해. 한쪽 철도엔 한 명의 사람이, 다른 철도엔 다섯 명의 사람이 묶여있어. 기차가 어느 쪽으로 가든 사람들이 사망하는 상황이지. 이런 상황에서 어떤 결정을 내려야 할지가 딜레마인거지.
'''
  output_file = "contents/tts.wav"
  
  TTS().text_to_speech(text, output_file)
  print(111)