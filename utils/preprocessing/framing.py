""" This function clips audio data and converts it into frames."""

import numpy as np

def Framing(signal, sample_rate, frame_size = 0.025, frame_stride = 0.01):
    signal = signal.to("cpu")
    signal = signal.numpy()
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = int(signal.size)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return num_frames, frames