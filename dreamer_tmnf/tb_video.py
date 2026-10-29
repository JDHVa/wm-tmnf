import os
import tempfile

import imageio
import numpy as np
import torch.utils.tensorboard.summary as tbs
from tensorboard.compat.proto.summary_pb2 import Summary


def make_video(tensor, fps):
    _, h, w, c = tensor.shape
    d = tempfile.mkdtemp()
    path = os.path.join(d, "v.gif")
    imageio.mimsave(path, list(np.asarray(tensor)), duration=1.0 / fps, loop=0)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    os.rmdir(d)
    return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=data)


def install():
    tbs.make_video = make_video
