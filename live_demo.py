import gradio as gr
from tools import Inference, Matting, log, extract_matte, harmonize, css, live_matting_step
from omegaconf import OmegaConf
import os
import sys
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image
import cv2 as cv
import time
import asyncio

args = OmegaConf.load(os.path.join(f"./config/test.yaml"))

log("Model loading")
phnet = Inference(**args)
stylematte = Matting(**args)
log("Model loaded")


async def show(queue):
    while True:
        log("SHOW FRAME")
        frame = queue.get()
        cv.imshow('Video', frame)
        await asyncio.sleep(0.01)


async def main(queue):
    video = cv.VideoCapture(0)
    fps = 10
    counter = 0
    frame_count = 0
    if not video.isOpened():
        raise Exception('Video is not opened!')
    begin = time.time()
    for i in range(300):
        counter += 1
        frame_count += 1
        ret, frame = video.read()  # Capture frame-by-frame
        inp = np.array(frame)
        back = np.zeros_like(frame)
        queue.put(inp)
        # res = asyncio.ensure_future(
        # live_matting_step(inp, back, stylematte))
        # res = await live_matting_step(inp, back, stylematte)
        log(f"{i} await")
        end = time.time()
        log(f"frames: {frame_count}, time: {end - begin}, fps: {frame_count/(end - begin) }")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    log(f"OVERALL TIME CONSUMED: {end - begin}, frames: {frame_count}, fps: {frame_count/(end - begin) }")
    # release the capture
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    # asyncio.ensure_future(show(frame))  # Display the resulting frame

    loop.run_until_complete(main(queue))
    loop.run_until_complete(show(queue))
    loop.run_forever()
