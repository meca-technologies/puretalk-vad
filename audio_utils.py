import numpy as np


def setup_audio_stream(audio, audio_format, channels, rate, chunk):
    """
    Set up an audio stream with the specified format, channels, rate, and chunk size.

    Parameters:
        audio: The audio module used to open the stream.
        audio_format: The format of the audio stream.
        channels (int): The number of audio channels.
        rate (int): The sampling rate in Hertz.
        chunk (int): The number of frames per buffer.

    Returns:
        stream: The opened audio stream.
    """
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    return stream


def int2float(sound):
    """
    Convert an array of integers representing sound samples to an array of floating point numbers.

    Parameters:
    sound (np.ndarray): The input array of sound samples.

    Returns:
    np.ndarray: The array of sound samples converted to floating point numbers.
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound
