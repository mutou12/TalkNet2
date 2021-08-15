import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Data_extra import dataset
from model import TalkNet2, GraphemeDuration, PitchPredictor
from configs import hyperparams as hp
from tqdm import tqdm
from glob import glob
import time
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_device(tensors: list, device='cuda:0'):
    return [tensor.to(device) for tensor in tensors]


# 整体训练
def train_eval():
    logs_idx = time.asctime(time.localtime(time.time())).replace(":", "_").replace("  ", " ").replace(" ", "_")
    writer = SummaryWriter(log_dir=f'logs/{logs_idx}')

    durmodel = GraphemeDuration(idim=hp.idim).to(device)
    pitchmodel = PitchPredictor(idim=hp.idim).to(device)
    model = TalkNet2(idim=hp.idim, postnet_layers=5).to(device)

    durmodel.train()
    pitchmodel.train()
    model.train()
    dur_optimizer = torch.optim.Adam(lr=1e-5, weight_decay=1e-6, eps=1e-8, params=durmodel.parameters())
    pitch_optimizer = torch.optim.Adam(lr=1e-5, weight_decay=1e-6, eps=1e-8, params=pitchmodel.parameters())
    optimizer = torch.optim.Adam(lr=1e-5, weight_decay=1e-6, eps=1e-8, params=model.parameters())

    epoch = 0
    train_global_idx = 0
    val_global_idx = 0

    trainDataSet = dataset.BZN_dataset(is_Train=True)
    evalDataSet = dataset.BZN_dataset(is_Train=False)

    for e in range(epoch, hp.epochs + 1):
        trainDataLoader = DataLoader(trainDataSet,
                                     batch_size=hp.batch_size,
                                     collate_fn=trainDataSet.collection,
                                     shuffle=True,
                                     num_workers=8)
        train_par = tqdm(trainDataLoader)

        for i, batch in enumerate(train_par):
            ids, raw_texts, texts, dur_texts, text_lens, max_text_lens, mels, mel_lens, max_mel_lens, f0, energies, \
            durations, f0_mask = batch
            texts, mels, f0, energies, durations, text_lens, f0_mask, mel_lens, dur_texts = \
                to_device([texts, mels, f0, energies, durations, text_lens, f0_mask, mel_lens, dur_texts])

            # 时间长度
            pred_durs = durmodel(texts, text_lens)
            durloss, acc, acc_dist_1, acc_dist_3 = durmodel._metrics(true_durs=durations, true_text_len=text_lens,
                                                                     pred_durs=pred_durs)

            # f0预测
            pred_f0_sil, pred_f0_body = pitchmodel(texts, durations)
            pitch_loss, sil_acc, body_mae = pitchmodel._metrics(
                true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
            )

            # mel图预测
            before_outs, after_outs = model(texts, durations, f0)
            loss_mel = model._metrics(mels, mel_lens, before_outs)
            loss_mel_post = model._metrics(mels, mel_lens, after_outs)
            loss = loss_mel_post + loss_mel

            train_par.set_description(
                f'🌏{e}, dur_loss={durloss:.5f}, dur_acc={acc:.5f}, dur_acc_dist_1={acc_dist_1:.5f}, '
                f'dur_acc_dist_3s={acc_dist_3:.5f}, '
                f'pitch_loss={pitch_loss:.5f}, sil_acc={sil_acc:.5f}, body_mae={body_mae:.5f}, '
                f'loss_mel={loss_mel:.5f}, loss_mel_post={loss_mel_post:.5f}')

            dur_optimizer.zero_grad()
            durloss.backward()
            torch.nn.utils.clip_grad_norm_(durmodel.parameters(), 1.0)
            dur_optimizer.step()

            pitch_optimizer.zero_grad()
            pitch_loss.backward()
            torch.nn.utils.clip_grad_norm_(pitchmodel.parameters(), 1.0)
            pitch_optimizer.step()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if train_global_idx % 200 == 0:
                writer.add_scalar('train_dur/dur_loss',
                                  durloss,
                                  train_global_idx)
                writer.add_scalars('train_dur/dur_acc',
                                   {'acc': acc, 'dur_acc_dist_1': acc_dist_1, 'dur_acc_dist_3s': acc_dist_3},
                                   train_global_idx)
                writer.add_scalar('train_pitch/pitch_loss',
                                  pitch_loss,
                                  train_global_idx)
                writer.add_scalar('train_pitch/sil_acc',
                                  sil_acc,
                                  train_global_idx)
                writer.add_scalar('train_pitch/body_mae',
                                  body_mae,
                                  train_global_idx)
                writer.add_scalars('train_mel/loss_mel',
                                   {'loss_mel': loss_mel, 'loss_mel_post': loss_mel_post},
                                   train_global_idx)

            if train_global_idx % 500 == 0:
                fig, ax = plt.subplots(3, 1)
                # add your subplots with some images eg.
                ax[0].imshow(before_outs[0].data.cpu().numpy())
                ax[1].imshow(after_outs[0].data.cpu().numpy())
                ax[2].imshow(mels[0].T.data.cpu().numpy())
                # etc.
                writer.add_figure("train", fig, train_global_idx)

            train_global_idx += 1
            torch.cuda.empty_cache()

        evalDataLoader = DataLoader(evalDataSet,
                                    batch_size=hp.batch_size,
                                    collate_fn=evalDataSet.collection,
                                    shuffle=True,
                                    num_workers=1)
        eval_par = tqdm(evalDataLoader)
        with torch.no_grad():
            durmodel.eval()
            pitchmodel.eval()
            model.eval()
            for i, batch in enumerate(eval_par):
                ids, raw_texts, texts, dur_texts, text_lens, max_text_lens, mels, mel_lens, max_mel_lens, f0, energies, \
                durations, f0_mask = batch
                texts, mels, f0, energies, durations, text_lens, f0_mask, mel_lens, dur_texts = \
                    to_device([texts, mels, f0, energies, durations, text_lens, f0_mask, mel_lens, dur_texts])

                pred_durs = durmodel(texts, text_lens)
                durloss, acc, acc_dist_1, acc_dist_3 = durmodel._metrics(true_durs=durations, true_text_len=text_lens,
                                                                         pred_durs=pred_durs)

                pred_f0_sil, pred_f0_body = pitchmodel(texts, durations)

                pitch_loss, sil_acc, body_mae = pitchmodel._metrics(
                    true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
                )

                before_outs, after_outs = model(texts, durations, f0)
                loss_mel = model._metrics(mels, mel_lens, before_outs)
                loss_mel_post = model._metrics(mels, mel_lens, after_outs)

                eval_par.set_description(
                    f'💙{e}, dur_loss={durloss:.5f}, dur_acc={acc:.5f}, dur_acc_dist_1={acc_dist_1:.5f}, '
                    f'dur_acc_dist_3s={acc_dist_3:.5f}, '
                    f'pitch_loss={pitch_loss:.5f},sil_acc={sil_acc:.5f}, body_mae={body_mae:.5f}，'
                    f'loss_mel={loss_mel:.5f}, loss_mel_post={loss_mel_post:.5f}')

                if val_global_idx % 7 == 0:
                    writer.add_scalar('val_dur/dur_loss',
                                      durloss,
                                      val_global_idx)
                    writer.add_scalars('val_dur/dur_acc',
                                       {'acc': acc, 'dur_acc_dist_1': acc_dist_1, 'dur_acc_dist_3s': acc_dist_3},
                                       val_global_idx)
                    writer.add_scalar('val_pitch/pitch_loss',
                                      pitch_loss,
                                      val_global_idx)
                    writer.add_scalar('val_pitch/sil_acc',
                                      sil_acc,
                                      val_global_idx)
                    writer.add_scalar('val_pitch/body_mae',
                                      body_mae,
                                      val_global_idx)
                    writer.add_scalars('val_mel/loss_mel',
                                       {'loss_mel': loss_mel, 'loss_mel_post': loss_mel_post},
                                       val_global_idx)

                if val_global_idx % 7 == 0:
                    fig, ax = plt.subplots(3, 1)
                    # add your subplots with some images eg.
                    ax[0].imshow(before_outs[0].data.cpu().numpy())
                    ax[1].imshow(after_outs[0].data.cpu().numpy())
                    ax[2].imshow(mels[0].T.data.cpu().numpy())
                    # etc.
                    writer.add_figure("val", fig, val_global_idx)

                val_global_idx += 1

        saves = glob(f'logs/{logs_idx}/*.pt')
        if len(saves) == 10:
            saves.sort(key=os.path.getmtime)
            os.remove(saves[0])

        if e % 50 == 0:
            torch.save({
                'epoch': e + 1,
                'global_idx': train_global_idx,
                'durmodel': durmodel.state_dict(),
                'pitchmodel': pitchmodel.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                f'logs/{logs_idx}/model_{e + 1}.pt')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train_eval()
