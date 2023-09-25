import onnx
from onnxruntime import InferenceSession
import numpy as np
import torch


def execute_onnx_model(x, onnx_model) -> None:
    sess = InferenceSession(onnx_model.SerializeToString(), providers=[
                            'AzureExecutionProvider', 'CPUExecutionProvider'])
    out = sess.run(None, {'input.1': x.numpy().astype(np.float32)})[0]

    return out
