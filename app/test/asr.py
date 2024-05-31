import speech_recognition as sr

r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        # Auto Speech Recognition
        audio = r.listen(source)
        user_text = None
        user_text = r.recognize_whisper(audio, language="japanese")
        print(user_text)