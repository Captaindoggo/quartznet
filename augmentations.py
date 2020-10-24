import numpy as np
import torch

from torch import nn
from torch import distributions

import librosa

import torchaudio




class stretcher(object):
    def __init__(self, range_low=0.7, range_high=1.3):
        self.low = range_low
        self.high = range_high

    def __call__(self, sample):
        wav = librosa.effects.time_stretch(sample.numpy(), np.random.uniform(self.low, self.high))
        wav = torch.from_numpy(wav)
        return wav


class pitch_shift(object):
    def __init__(self, range_low=-4, range_high=4, sr=16000):
        self.low = range_low
        self.high = range_high
        self.sr = sr

    def __call__(self, sample):
        wav = librosa.effects.pitch_shift(sample.numpy(), self.sr, np.random.uniform(self.low, self.high))
        wav = torch.from_numpy(wav)
        return wav


class noizer(object):
    def __init__(self, m=0, d=0.0005):
        self.m = m
        self.d = d

    def __call__(self, sample):
        noiser = distributions.Normal(self.m, self.d)
        wav = sample + noiser.sample(sample.size())
        wav.clamp_(-1, 1)
        return wav


def my_collate(data):
    max_l = 0
    max_wl = 0
    for i in data:
        if len(i[0]) > max_l:
            max_l = len(i[0])
        if len(i[1]) > max_wl:
            max_wl = len(i[1])
    wavs = []
    spec_lens = []
    ans_lens = []
    answers = []
    for i in range(len(data)):
        audio = data[i][0]
        spec_lens.append((len(audio) - 1024) // 256 + 5)
        ans_lens.append(len(data[i][1]))
        ansr = torch.tensor(data[i][1])

        if len(audio) < max_l:
            audio = torch.cat((audio, torch.zeros(max_l - len(audio))))
        wavs.append(audio)
        if len(ansr) < max_wl:
            ansr = torch.cat((ansr, torch.zeros(max_wl - len(ansr))))
        answers.append(ansr)

    batch_wavs = torch.stack(wavs)
    batch_ans = torch.stack(answers)

    batch_ans_lens = torch.tensor(ans_lens)
    batch_spec_lens = torch.tensor(spec_lens)

    return batch_wavs, batch_ans, batch_spec_lens, batch_ans_lens





class LogMelSpectrogram(nn.Module):

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64):
        super(LogMelSpectrogram, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256, f_min=0, f_max=8000)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(waveform).squeeze()
        return torch.log(spectrogram + 1e-9)