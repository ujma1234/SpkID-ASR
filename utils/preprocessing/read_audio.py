import soundfile as sf
import torchaudio

def with_soundfile(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3

def with_torchaudio(fname, sr = 44100):
    audio_data, sample_rate = torchaudio.load(fname, sr)
    return audio_data, sample_rate
