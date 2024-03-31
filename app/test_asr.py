import speech_recognition as sr
#import sys #-- 텍스트 저장시 사용

r = sr.Recognizer()
kr_audio = sr.AudioFile('./test_file.wav')

with kr_audio as source:
    audio = r.listen(source)

#sys.stdout = open('news_out.txt', 'w') #-- 텍스트 저장시 사용
print(r.recognize_google(audio, language='ko-KR'))