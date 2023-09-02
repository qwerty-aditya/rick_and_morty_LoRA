# Text to Image LoRA training to generate Rick and Morty art style Characters

### Training Process:
1. Fine Tuning the Stable diffusion model with the LoRA method is very fast and efficient.
2. The train_text_to_image_lora library provided by Huggingface was used for the fine-tuning purposes.
3. Created a Rick and Morty Dataset of 40 images: LINK
4. Some of the important Hyperparameters used during the fine-tuning were:
   1. Learning Rate = 1e-04
   1. Training steps = 2000
   1. Image resolution = 512
5. This fine tuned model had a safetensor output file which contained the newly trained weights.
6. These weights were added on top of the ‘runwayml/stable-diffusion-v1-5’
7. The final model was then used for the text-to-image inferencing.

### Steps to Launch the WebApp:
>Note: Since Inferencing required Server with Nvidia GPU, which has a huge cost associated with it, Google Colab GPU was used for the deployment.
[Notebook Link](https://colab.research.google.com/drive/10pos1pk3Cg2wNO0rBgbTbtdDCZ5bJ7lL#scrollTo=V17OK-Wuognf)

1. Run all the cells in the Notebook, this will install the requirements and deploy the web app, finally grok will make the Web App available to the internet.
2. Please click the ngrok-free.app link in the output of the last cell to launch the Web App.

![Screenshot of Web App](rick_and_morty_LoRA/Screenshot 2023-09-02 at 7.10.58 PM.png)
