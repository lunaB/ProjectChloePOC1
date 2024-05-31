from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

import dotenv
import os
dotenv.load_dotenv()


HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

model_id = "google/paligemma-3b-mix-224"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)


from PIL import Image

im = Image.open("contents/screen.png")
rgb_im = im.convert('RGB')
rgb_im.save('contents/screen.jpg')

url = "contents/screen.jpg"
image = Image.open(url)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, token=HUGGING_FACE_API_KEY).eval()
processor = AutoProcessor.from_pretrained(model_id, token=HUGGING_FACE_API_KEY)


def generate_caption(prompt, image: Image) -> str:
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

print(generate_caption("Describe what is happening on the screen : ", image))
import pdb; pdb.set_trace()

