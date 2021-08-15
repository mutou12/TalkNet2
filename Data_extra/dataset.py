from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import tgt
import processing
from tqdm import tqdm
import text
import collections
from configs.tools import pad_1D, pad_2D
import torch


class BZN_dataset(Dataset):
    def __init__(self, is_Train=True):
        wavepath = r'F:\data_BZN\sox\*.wav'
        if is_Train:
            self.waveArr = glob(wavepath)[:9900]
        else:
            self.waveArr = glob(wavepath)[9900:]

    def __len__(self):
        return len(self.waveArr)

    def __getitem__(self, item):
        index = self.waveArr[item].split('\\')[-1].split('.')[0]
        # if index == '000611':
        #     index = '001001'
        textgird_path = 'F:\data_BZN\PhoneLabeling'
        tgrid = tgt.io.read_textgrid(textgird_path + f'\{index}.interval')
        phone, duration, start, end = processing.get_alignment(tgrid.tiers[0])
        #  phone, duration, start, end = processing.get_alignment_000611(tgrid.tiers[0])
        phon = "{" + " ~ ".join(phone) + " ~}"
        phon = np.array(text.text_to_sequence(phon, ['transliteration_cleaners']))

        # if len(phone) != len(phon):
        # print(index, len(phone), len(phon), len(duration), phone, phon)
        duration = sum([[1 if i > 0 else i, i - 1 if i - 1 > 0 else 0] for i in duration], [])

        dur_text = []
        for t, d in zip(phon, duration):
            for i in range(d):
                dur_text.append(t)
        dur_text = np.array(dur_text)
        if start >= end:
            return None
        mel = np.load("{}/{}.npy".format('F:/data_BZN/sox/', index + '_mel'))
        energy = np.load("{}/{}.npy".format('F:/data_BZN/sox/', index + '_e'))
        f0 = np.load("{}/{}.npy".format('F:/data_BZN/sox/', index + '_p'))

        with open("{}/{}.txt".format('F:/data_BZN/Wave/', index), "r") as f:  # 打开文件
            raw_text = f.readline().strip("\n")  # 读取文件
        raw_text += '|'
        raw_text += ' '.join(phone).replace('sp1', '')
        # if sum(duration) > len(f0):
        #     print(index)
        sample = {
            "id": index,
            "text": phon,
            "raw_text": raw_text,
            "dur_text": dur_text,
            "mel": mel.T[:sum(duration), :],
            "f0": f0[: sum(duration)],
            "energy": energy[: sum(duration)],
            "duration": duration,
            "textlength": len(phon)
        }
        return sample

    def make_mask(self, lengths, max_length=None):
        """Makes mask from list of lengths."""
        device = lengths.device if torch.is_tensor(lengths) else 'cpu'
        lengths = lengths if torch.is_tensor(lengths) else torch.tensor(lengths)
        max_length = max_length or torch.max(lengths)
        start = torch.tensor(0).int()
        indices = torch.arange(start=start, end=max_length, device=device)  # noqa
        mask = indices.lt(lengths.view(-1, 1))

        return mask

    def collection(self, batch):
        if isinstance(batch[0], collections.Mapping):
            ids = [d['id'] for d in batch]
            texts = [d['text'] for d in batch]
            raw_texts = [d['raw_text'] for d in batch]
            f0 = [d['f0'] for d in batch]
            mels = [d['mel'] for d in batch]
            energies = [d['energy'] for d in batch]
            dur_texts = [d['dur_text'] for d in batch]
            durations = [np.array(d['duration']) for d in batch]
            text_length = [d['textlength'] for d in batch]

            ids = [i for i, _ in sorted(zip(ids, text_length), key=lambda x: x[1], reverse=True)]
            texts = [i for i, _ in sorted(zip(texts, text_length), key=lambda x: x[1], reverse=True)]
            raw_texts = [i for i, _ in sorted(zip(raw_texts, text_length), key=lambda x: x[1], reverse=True)]
            dur_texts = [i for i, _ in sorted(zip(dur_texts, text_length), key=lambda x: x[1], reverse=True)]
            f0 = [i for i, _ in sorted(zip(f0, text_length), key=lambda x: x[1], reverse=True)]
            mels = [i for i, _ in sorted(zip(mels, text_length), key=lambda x: x[1], reverse=True)]
            energies = [i for i, _ in sorted(zip(energies, text_length), key=lambda x: x[1], reverse=True)]
            durations = [i for i, _ in sorted(zip(durations, text_length), key=lambda x: x[1], reverse=True)]

            text_lens = np.array([text.shape[0] for text in texts])
            mel_lens = np.array([mel.shape[0] for mel in mels])

            texts = pad_1D(texts).astype(np.int32)
            dur_texts = pad_1D(dur_texts).astype(np.int32)
            mels = pad_2D(mels)
            f0_mask = self.make_mask([f.shape[-1] for f in f0])  # noqa
            f0 = pad_1D(f0)
            energies = pad_1D(energies)
            durations = pad_1D(durations)

            texts = torch.LongTensor(texts)
            dur_texts = torch.LongTensor(dur_texts)
            text_lens = torch.LongTensor(text_lens)
            mels = torch.FloatTensor(mels)
            mel_lens = torch.LongTensor(mel_lens)
            pitches = torch.FloatTensor(f0)
            f0_mask = torch.BoolTensor(f0_mask)
            energies = torch.LongTensor(energies)
            durations = torch.LongTensor(durations)

            return (
                ids,
                raw_texts,
                texts,
                dur_texts,
                text_lens,
                max(text_lens),
                mels,
                mel_lens,
                max(mel_lens),
                pitches,
                energies,
                durations,
                f0_mask
            )


def Pre_progress_BZN_data():
    trainDataSet = BZN_dataset(is_Train=False)
    trainDataLoader = DataLoader(trainDataSet, batch_size=1,
                                 num_workers=3, collate_fn=trainDataSet.collection)
    par = tqdm(trainDataLoader)

    z = []

    # removesil(par)
    for i, f in enumerate(par):
        m = 0
        n = m
        for p in f[1][0].split('|')[0].split(' '):
            while p[-1].isdigit() != f[1][0].split('|')[1][m].isdigit():
                m += 1
            z.append(p[:-1] + ' ' + f[1][0].split('|')[1][n: m])
            m += 2
            n = m

    print(np.unique(z))

    fh = open(r"E:\TalkNet2\text\pinyin2phone.txt", 'w', encoding='utf-8')
    for zz in np.unique(z):
        fh.write(zz)
    fh.close()


class ASHILL_dataset(Dataset):
    def __init__(self, is_Train=True):
        wavepath = r'F:\data_BZN\sox\*.wav'
        if is_Train:
            self.waveArr = glob(wavepath)[:9900]
        else:
            self.waveArr = glob(wavepath)[9900:]
        self.waveArr = glob(wavepath)

    def __len__(self):
        return len(self.waveArr)

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    trainDataSet = ASHILL_dataset(is_Train=False)
    trainDataLoader = DataLoader(trainDataSet, batch_size=1,
                                 num_workers=3, collate_fn=trainDataSet.collection)
    par = tqdm(trainDataLoader)

    for i, f in enumerate(par):
        pass
