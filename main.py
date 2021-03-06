import math
import time

import numpy as np
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k

from model import *
from utils import *

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

N_EPOCHS = 10
CLIP = 1


def run(model, SRC, TRG, train_data, valid_data, test_data, model_name="nmt-model-attn.pt"):
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    model_summary = model.apply(init_weights)
    print(model_summary)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'\t Test. Loss: {test_loss:.3f}')
    cal_bleu_score(train_data, model, SRC, TRG, device)


if __name__ == '__main__':
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)  # chi cho phep cac token xuat hien it nhat 2 lan lam tu vung
    TRG.build_vocab(train_data, min_freq=2)

    INPUT_DIM = 7855  # len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load("nmt-model-attn.pt", map_location=device))
    model.eval()
    display_attention(" ".join(train_data.examples[1011].src), model, SRC, TRG, device)
    # run(model, SRC, TRG, train_data, valid_data, test_data)
    #
    # enc_rnn = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    # dec_rnn = DecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)
    #
    # model_rnn = Seq2SeqRNN(enc_rnn, dec_rnn, device).to(device)
    # run(model_rnn, SRC, TRG, train_data, valid_data, test_data, "nmt-model-base.pt")

    # model.load_state_dict(torch.load("nmt-model-rnn.pt", map_location=device))
    # model.eval()
