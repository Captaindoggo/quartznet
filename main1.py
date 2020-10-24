import numpy as np
import pandas as pd
import torch

from nemo.core.optim.optimizers import Novograd
import wandb


from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample


from torch import nn


import string
import random

from sklearn.model_selection import train_test_split


from Qartznet import QNet
from encoding import decode_wer
from augmentations import noizer, pitch_shift, stretcher, LogMelSpectrogram, my_collate
from dataset_loader import common_voice_dataset



def set_seed(n):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed = n
    random.seed(n)
    np.random.seed(n)





if __name__ == '__main__':
    set_seed(42)

    wandb.init(project="dla-homework-2")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dirr = 'cv-corpus-5.1-2020-06-22/en/train.tsv'
    data = pd.read_csv(dirr, sep='\t')

    X = data['path']
    y = data['sentence']

    X = np.array(X)
    y = np.array(y)

    seq_lens = list(map(lambda x: len(x), y))
    sorted_idxs = np.argsort(seq_lens)
    X = X[sorted_idxs]
    y = y[sorted_idxs]

    # cut_idxs = int(0.35 * len(X))
    cut_idxs = 120000

    X = X[0:cut_idxs]
    y = y[0:cut_idxs]

    remove = set(string.punctuation)

    word_to_idx = {}

    word_to_idx[' '] = 0
    word_to_idx['"'] = 1
    ctr = 2
    for i in list(string.ascii_lowercase):
        word_to_idx[i] = ctr
        ctr += 1

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    encoded_y = []

    mistakes = set()
    for i in range(len(y)):
        sentence = y[i].lower()
        encod = []
        for j in sentence:
            if j not in remove:
                if j in set(word_to_idx.keys()):
                    encod.append(word_to_idx[j])
                else:
                    mistakes.add(i)
        encoded_y.append(encod)

    encoded_y = np.array(encoded_y)
    clean_idxs = list(set(np.arange(0, len(y))) - set(mistakes))

    clean_encoded_y = encoded_y[clean_idxs]
    clean_X = X[clean_idxs]

    dirr = 'cv-corpus-5.1-2020-06-22/en/clips/'

    X_train, X_test, y_train, y_test = train_test_split(clean_X, clean_encoded_y, test_size=0.2)

    train_dataset = common_voice_dataset(X_train, y_train, dirr,
                                         transform=[Resample(orig_freq=48000, new_freq=16000),
                                                    pitch_shift(), stretcher(), noizer()])

    val_dataset = common_voice_dataset(X_test, y_test, dirr,
                                         transform=[Resample(orig_freq=48000, new_freq=16000)])

    batch_size = 80
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, collate_fn=my_collate)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                               shuffle=False, collate_fn=my_collate)

    sr = 16000
    n_mels = 64

    lr = 0.1

    n_epochs = 6

    start_epoch = -1

    n_classes = len(word_to_idx)

    model = QNet(n_mels, n_classes).to(device)

    if start_epoch != -1:
        model_saved = 'checkpoint' + str(start_epoch-1) + '.pt'
        model.load_state_dict(torch.load(
            model_saved,
            map_location=device))

    #print("Total number of trainable parameters:", count_parameters(model))

    optimizer = Novograd(model.parameters(), lr=lr, betas=(0.8, 0.5), weight_decay=0.01)

    if start_epoch != -1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-start_epoch, eta_min=0, last_epoch=-1)

        prev_val_loss = 575.588
        scheduler.step(prev_val_loss)

        for i in optimizer.param_groups:
            print(i['lr'])

    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0, last_epoch=-1)

    criterion = nn.CTCLoss()

    melspec = LogMelSpectrogram().to(device)


    wandb.watch(model)

    for epoch in range(max(start_epoch, 0), n_epochs):
        running_loss = 0.0
        val_loss = 0.0
        model.train()
        b_ctr = 0
        for batch in train_loader:
            X, y, x_len, y_len = batch
            X = X.to(device)
            y = y.to(device)
            x_len = x_len.to(device)
            y_len = y_len.to(device)
            optimizer.zero_grad()


            output = melspec(X)

            output = model(output)

            output_T = output.transpose(1, 2).transpose(0, 1)
            loss = criterion(output_T, y, x_len//2, y_len)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if b_ctr % 100 == 0:
                print('batch:', b_ctr)
            b_ctr+=1

        ch_pt = './checkpoint' + str(epoch) + '.pt'

        torch.save(model.state_dict(), ch_pt)

        print('epoch', epoch, 'weights saved')

        preds = []
        true = []

        for batch in val_loader:
            X, y, x_len, y_len = batch
            model.eval()
            with torch.no_grad():
                X = X.to(device)
                y = y.to(device)
                x_len = x_len.to(device)
                y_len = y_len.to(device)

                pred = melspec(X)

                pred = model(pred)

                pred_T = pred.transpose(1, 2).transpose(0, 1)
                loss = criterion(pred_T, y, x_len // 2, y_len)
            val_loss += loss.item()

            pred = torch.argmax(pred, 1)

            preds.append(pred.to('cpu'))
            true.append(y.to('cpu'))

        wer = decode_wer(preds, true, idx_to_word)

        # print('epoch:', epoch + 1, 'train loss:', running_loss, 'val loss:', val_loss, 'wer:', wer)

        wandb.log({"Train Loss": running_loss, "Val Loss": val_loss, "Val WER:": wer})

        scheduler.step(val_loss)




