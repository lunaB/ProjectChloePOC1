from googletrans import Translator

translator = Translator()
text = "誰のことを指してるの？ちょっと詳しく教えて！"
a = translator.translate(text, dest='ko')
print(a)