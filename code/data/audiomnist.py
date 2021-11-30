import torch
import torch.nn as nn
import glob
import math
import decimal
import os

import numpy as np
import librosa as li

from os import path


###############################################################################
# Cache and load AudioMNIST dataset
###############################################################################


class AudioMNISTDataset(torch.utils.data.Dataset):

    def __init__(self,
                 split: str = 'test',
                 sample_rate: int = 16000,
                 signal_length: float = 1.0,
                 scale: float = 1.0):

        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split {split}; must be one of "train" or'
                             f' "test"')

        self.sample_rate = sample_rate
        self.signal_length = math.floor(signal_length * self.sample_rate)
        self.scale = scale
        self.ext = 'wav'

        self.data_dir = path.join(
            os.fspath(AUDIOMNIST_DATA_DIR),
            split
        )
        self.cache_dir = path.join(
            os.fspath(AUDIOMNIST_CACHE_DIR),
            split
        )

        ensure_dir(self.data_dir)
        ensure_dir(self.cache_dir)

        self.audio_list = sorted(list(Path(self.data_dir).rglob(f'*.{self.ext}')))

        cache_list = sorted(list(Path(self.cache_dir).rglob('*.pt')))

        if len(cache_list) > 0:
            self.tx = torch.load(path.join(self.cache_dir, 'tx.pt'))
            self.ty = torch.load(path.join(self.cache_dir, 'ty.pt'))

        else:
            self.tx, self.ty = self._build_cache()

        self.n_classes = torch.unique(self.ty).shape[-1]

        # register data properties
        DataProperties.register(
            sample_rate=self.sample_rate,
            signal_length=self.signal_length,
            scale=self.scale,
            n_classes=self.n_classes
        )

    def _build_cache(self):
        # cache dataset in tensor form
        tx = torch.zeros((len(self.audio_list), 1, self.signal_length))
        ty = torch.zeros(len(self.audio_list), dtype=torch.long)

        pbar = tqdm(self.audio_list,
                    total=len(self.audio_list))
        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading AudioMNIST ({path.basename(audio_fn)})')
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=self.sample_rate,
                                  duration=self.signal_length)
            waveform = torch.from_numpy(waveform)

            tx[i, :, :waveform.shape[-1]] = waveform
            ty[i] = int(path.basename(audio_fn).split("_")[0])

        # apply scale
        tx *= self.scale

        torch.save(tx, path.join(self.cache_dir, 'tx.pt'))
        torch.save(ty, path.join(self.cache_dir, 'ty.pt'))

        return tx, ty

    def __len__(self):
        return self.tx.shape[0]

    def __getitem__(self, idx):

        sample = {
            'input': self.tx[idx],
            'digit': self.ty[idx]
        }

        return sample















