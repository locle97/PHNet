import gradio as gr
from tools import Inference, Matting, log, extract_matte, harmonize, css
from omegaconf import OmegaConf
import os
import sys
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image

args = OmegaConf.load(os.path.join(f"./config/demo.yaml"))

log("Model loading")
phnet = Inference(**args)
stylematte = Matting(**args)
log("Model loaded")

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Welcome to portrait transfer demo app!
    Select source portrait image and new background.
    """)
    btn_compose = gr.Button(value="Compose")

    with gr.Row():
        input_ui = gr.Image(
            type="numpy", label='Source image to extract foreground')
        back_ui = gr.Image(type="pil", label='The new background')

    gr.Examples(
        examples=[["./assets/comp.jpg", "./assets/back.jpg"]],
        inputs=[input_ui, back_ui],
    )

    gr.Markdown(
        """
    ## Resulting alpha matte and extracted foreground.
    """)
    with gr.Row():
        matte_ui = gr.Image(type="pil", label='Alpha matte')
        fg_ui = gr.Image(type="pil", image_mode='RGBA',
                         label='Extracted foreground')

    gr.Markdown(
        """
    ## Click the button and compare the composite with the harmonized version.
    """)
    btn_harmonize = gr.Button(value="Harmonize composite")

    with gr.Row():
        composite_ui = gr.Image(type="pil", label='Composite')
        harmonized_ui = gr.Image(
            type="pil", label='Harmonized composite', css=css(3, 3))

    btn_compose.click(lambda x, y: extract_matte(x, y, stylematte), inputs=[input_ui, back_ui], outputs=[
                      composite_ui, matte_ui, fg_ui])
    btn_harmonize.click(lambda x, y: harmonize(x, y, phnet), inputs=[
                        composite_ui, matte_ui], outputs=[harmonized_ui])


log("Interface created")
demo.launch(share=False)
