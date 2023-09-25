from multiprocessing import Process, Queue
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


def show(queue, stack):
    print(f"PROCESS {3}")
    # while not queue.empty():
    if stack.empty():
        frame = queue.get()
    else:
        frame = stack.get(block=False)
    cv.imshow('Video', np.uint8(frame))
    log("PID: 3, SHOW FRAME")
    print(frame.shape)
    time.sleep(0.1)


def extract(queue, stack, model):
    '''
    img: np.array, 
    back: np.array,
    model: Matting instance
    '''
    print(f"PROCESS {2}")
    img = queue.get()
    back = np.zeros_like(img)
    mask, fg = model.extract(img)
    composite = fg + (1 - mask[:, :, None]) * \
        back  # .resize(mask.shape[::-1])
    stack.put(np.uint8(composite))
    # time.sleep(0.1)
    print("PID: 2, LIVE STEP")
    # for i in range(10):
    #     print(f"In live {i}")
    # cv.imshow('Video', np.uint8(composite))
    # return composite


def main(queue):

    log(f"PROCESS {1}")
    video = cv.VideoCapture(0)
    fps = 10
    counter = 0
    frame_count = 0
    if not video.isOpened():
        raise Exception('Video is not opened!')
    begin = time.time()
    # stack = Queue()
    for i in range(10):
        counter += 1
        frame_count += 1
        ret, frame = video.read()  # Capture frame-by-frame
        inp = np.array(frame)
        back = np.zeros_like(frame)
        # res = asyncio.ensure_future(
        # live_matting_step(inp, back, stylematte))
        # res = live_matting_step(inp, back, stylematte)
        queue.put(inp)
        mp.sleep(0.1)
        # Display the resulting frame

        # blurred_frame = cv.blur(frame, (10, 10))
        counter = 0
        end = time.time()
        log(f"PID: 1, frames: {frame_count}, time: {end - begin}, fps: {frame_count/(end - begin) }")
    # else:
        # show(queue)  # Display the resulting frame

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    log(f"OVERALL TIME CONSUMED: {end - begin}, frames: {frame_count}, fps: {frame_count/(end - begin) }")
    # release the capture
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    queue = Queue()  # Создаем канал
    stack = Queue()  # Создаем канал
    # stack = Queue()  # Создаем канал
    args = OmegaConf.load(os.path.join(f"./config/test.yaml"))

    log("Model loading")
    phnet = Inference(**args)
    stylematte = Matting(**args)
    log("Model loaded")

    p1 = Process(target=main, args=(queue,))  # Вводим параметры

    p2 = Process(target=extract, args=(
        queue, stack, stylematte))  # Вводим параметры
    p3 = Process(target=show, args=(queue, stack))  # Вводим параметры
    # p2 = Process(target=test_2, args=("Пончик", queue,))  # Вводим параметры

    p1.start()
    p2.start()
    p3.start()
    p3.join()
    p2.join()
    p1.join()
