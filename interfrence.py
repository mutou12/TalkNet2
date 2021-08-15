from model import GraphemeDuration, PitchPredictor, TalkNet2
import torch
import hyperparams as hp
import numpy as np
import text

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def text2vctors(phon):
    phon = "{" + " ~ ".join(phon) + " ~}"
    phon = np.array(text.text_to_sequence(phon, ['transliteration_cleaners']))
    return torch.LongTensor([phon]).to(device), torch.IntTensor([len(phon)]).to(device)


def forward_for_export(tokens: torch.Tensor, text_len: torch.Tensor, durCp, pitchCp, modelCp):
    durmodel = GraphemeDuration(idim=hp.idim).to(device)
    pitchmodel = PitchPredictor(idim=hp.idim).to(device)
    model = TalkNet2(idim=hp.idim, postnet_layers=5).to(device)

    durmodel.load_state_dict(torch.load(durCp)['durmodel'])
    durmodel.eval()

    pitchmodel.load_state_dict(torch.load(pitchCp)['pitchmodel'])
    pitchmodel.eval()

    model.load_state_dict(torch.load(modelCp)['model'])
    model.eval()

    durs = durmodel(tokens, text_len)
    durs = durs.exp() - 1
    durs[durs < 0.0] = 0.0
    durs = durs.round().long()

    # Pitch
    f0_sil, f0_body = pitchmodel(tokens, durs)
    sil_mask = f0_sil.sigmoid() > 0.5
    f0 = f0_body * hp.f0_std + hp.f0_mean
    f0 = (~sil_mask * f0).float()

    # Spect
    _, mel = model(tokens, durs, f0)

    return mel


if __name__ == '__main__':
    path = r'E:\TalkNet2\Trainorinference\logs\Sat_Aug_7_14_15_33_2021\model_401.pt'
    _path = r'E:\TalkNet2\Trainorinference\logs\Sat_Aug_7_14_15_33_2021\model_101.pt'
    phon = ['uen2', 'j', 'ian4', 'k', 'e3', 'zh', 'iii2', 'j', 'ie1', 'sh', 'ang4', 'ch', 'uan2', 'sp1',
            'uen2', 'd', 'ang3', 'sh', 'u4', 'j', 'v5', 'j', 'ian4', 'i4', 'd', 'a3', 'b',
            'ao1', 'j', 'ia1', 'm', 'i4', 'h', 'ou4', 'sh', 'ang4', 'ch', 'uan2', 'sp1', 'm', 'i4', 'm',
            'a3', 't', 'ong1', 'g', 'uo4', 'q', 'i2', 't', 'a1', 't', 'u2', 'j', 'ing4', 'sp1', 'r', 'u2', 'uei1',
            'x', 'in4', 'f', 'a1', 's', 'ong4', 'sp1']

    tokens, text_len = text2vctors(phon)
    mel = forward_for_export(tokens, text_len, _path, _path, path)
    # torch.save(mel[0], r'E:\hifi-gan-master\test_mel_files\test1.wav.npy')
    np.save(r'E:\hifi-gan-master\test_mel_files\test1.wav.npy', mel.cpu().detach().numpy())
