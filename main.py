from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import base64
from io import BytesIO

# Load model
model_base = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
lora_model_path = "pytorch_lora_weights.safetensors"
pipe.unet.load_attn_procs(lora_model_path)
# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image = pipe("man on an aeroplane", num_inference_steps=50, guidance_scale=7.5, cross_attention_kwargs={"scale": 1}).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str)


if __name__ == '__main__':
    app.run()