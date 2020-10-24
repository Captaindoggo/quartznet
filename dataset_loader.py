from torch.utils.data import Dataset
import os
import torchaudio
import torch
from torchaudio import transforms


class common_voice_dataset(Dataset):

    def __init__(self, file_names, labels, root_dir, transform=[]):
        self.labels = labels
        self.file_names = file_names
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                  self.file_names[idx])
        audio = torchaudio.load(audio_name)[0][0]
        label = self.labels[idx]

        if len(self.transform) != 0:
            for trfm in self.transform:
                audio = trfm(audio)

        sample = [audio, label]

        return sample