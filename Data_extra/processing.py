import os

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from glob import glob
from configs import hyperparams as hp

# 小小驱动器
from stft import TacotronSTFT


class BZNSYP(Dataset):
    def __init__(self):
        wavepath = r'F:\data_BZN\sox\*.wav'
        self.waveArr = glob(wavepath)

    def __len__(self):
        return len(self.waveArr)

    def __getitem__(self, item):
        return self.waveArr[item]


from tqdm import tqdm
import wav_utils


# 提取mel谱图
def wav2mel(par):
    def wav2mel(par):
        stft = TacotronSTFT(
            filter_length=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            n_mel_channels=hp.n_mels,
            sampling_rate=hp.sample_rate,
            mel_fmin=hp.fmin,
            mel_fmax=hp.fmax,
        )

        for i, f in enumerate(par):
            par.set_description(f'{f[0]}')
            sr, wav = wav_utils.read_wav_np(f[0], hp.sample_rate)
            textgird_path = 'F:\data_BZN\PhoneLabeling'
            index = f[0].split('\\')[-1].split('.')[0]
            tgrid = tgt.io.read_textgrid(textgird_path + f'\{index}.interval')
            _, _, start, end = get_alignment(tgrid.tiers[0])
            wav = wav[
                  int(hp.sample_rate * start): int(hp.sample_rate * end)
                  ].astype(np.float32)
            p = wav_utils.pitch(wav, hp)
            wav = torch.from_numpy(wav).unsqueeze(0)
            mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
            mel = mel.squeeze(0)  # [num_mel, T]
            mag = mag.squeeze(0)  # [num_mag, T]
            e = torch.norm(mag, dim=0)  # [T, ]
            p = p[: mel.shape[1]]
            id = os.path.basename(f[0]).split(".")[0]
            np.save("{}/{}.npy".format('F:/data_BZN/sox/', id + '_mel'), mel.numpy(), allow_pickle=False)
            np.save("{}/{}.npy".format('F:/data_BZN/sox/', id + '_e'), e.numpy(), allow_pickle=False)
            np.save("{}/{}.npy".format('F:/data_BZN/sox/', id + '_p'), p, allow_pickle=False)


# 截取静默
def removesil(par):
    textgird_path = 'F:\data_BZN\PhoneLabeling'
    for i, f in enumerate(par):
        index = f[0].split('\\')[-1].split('.')[0]
        print(f[0].split('\\')[-1].split('.')[0])
        tgrid = tgt.io.read_textgrid(textgird_path + f'\{index}.interval')
        # tgrid = tgt.io.read_textgrid(filename=textgird_path + f'\BZT-{index}.TextGrid')
        l = get_alignment(tgrid.tiers[0])
        print(tgrid)


import tgt


def get_alignment(tier):
    sil_phones = ["sil", "sp1", "spn", ""]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if not phones:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * hp.sample_rate / hp.hop_length)
                - np.round(s * hp.sample_rate / hp.hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def get_alignment_000611(tier):
    sil_phones = ["sil", "sp1", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if not phones:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * hp.sample_rate / hp.hop_length)
                - np.round(s * hp.sample_rate / hp.hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


class AISHELL(Dataset):
    def __init__(self, is_Train=True):
        self.waveArr = []
        train_path = r'F:\AISHELL-3\train\wav'
        test_path = r'F:\AISHELL-3\test\wav'
        if is_Train:
            train_arr = glob(train_path + '\*')
            for f in train_arr:
                for ff in glob(f + '\*.wav'):
                    self.waveArr.append(ff)
        else:
            test_arr = glob(test_path + '\*')
            for f in test_arr:
                for ff in glob(f + '\*.wav'):
                    self.waveArr.append(ff)

    def __len__(self):
        return len(self.waveArr)

    def __getitem__(self, item):
        return self.waveArr[item]


if __name__ == '__main__':
    trainDataSet = AISHELL(is_Train=False)
    trainDataLoader = DataLoader(trainDataSet, batch_size=1,
                                 num_workers=1)
    par = tqdm(trainDataLoader)

    stft = TacotronSTFT(
        filter_length=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mel_channels=hp.n_mels,
        sampling_rate=hp.sample_rate,
        mel_fmin=hp.fmin,
        mel_fmax=hp.fmax,
    )

    # removesil(par)
    for i, f in enumerate(par):
        par.set_description(f'{f[0]}')

        path_arr = f[0].split('\\')
        index = path_arr[-1].split('.')[0]
        textgird_path = f'F:\data_AISHELL\phone\{path_arr[-2]}\{index}.TextGrid'

        # with open(textgird_path, 'r') as f:
        #     str1 = f.read().replace('""', '"sp1"')
        #     with open(textgird_path, 'w') as ff:
        #         ff.write(str1)

        sr_22050_train_path = 'F:\data_AISHELL\Wave\Train'
        sr_22050_val_path = 'F:\data_AISHELL\Wave\Val'
        if 'train' in f[0]:
            wav_path = sr_22050_train_path + f'\{index}.wav'
        elif 'test' in f[0]:
            wav_path = sr_22050_val_path + f'\{index}.wav'
        sr, wav = wav_utils.read_wav_np(wav_path, hp.sample_rate)
        textgird_path = f'F:\data_AISHELL\phone\{path_arr[-2]}\{index}.TextGrid'
        tgrid = tgt.io.read_textgrid(textgird_path)
        _, _, start, end = get_alignment(tgrid.tiers[1])
        wav = wav[
              int(hp.sample_rate * start): int(hp.sample_rate * end)
              ].astype(np.float32)
        p = wav_utils.pitch(wav, hp)
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]
        e = torch.norm(mag, dim=0)  # [T, ]
        p = p[: mel.shape[1]]
        id = os.path.basename(f[0]).split(".")[0]
        np.save("{}/{}.npy".format('F:/data_AISHELL/fme/', id + '_mel'), mel.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format('F:/data_AISHELL/fme/', id + '_e'), e.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format('F:/data_AISHELL/fme/', id + '_p'), p, allow_pickle=False)
