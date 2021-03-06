import de_core_news_sm
import en_core_web_sm
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import bleu_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

spacy_de = de_core_news_sm.load()
spacy_en = en_core_web_sm.load()


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def translate(model, sentence, src, trg, device, max_length=512):
    encoder, decoder = model.encoder, model.decoder
    sentence = sentence.lower()
    with torch.no_grad():
        input_tensor = [src.vocab.stoi[word] for word in sentence.split(' ')]
        input_tensor.append(src.vocab.stoi[src.eos_token])
        input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device).view(-1, 1)

        encoder_outputs, hidden = encoder(input_tensor)

        decoder_input = torch.tensor([trg.vocab.stoi[trg.init_token]], device=device)

        decoder_hidden = hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, 1, input_tensor.shape[0]).to(device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == trg.vocab.stoi[trg.eos_token]:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(trg.vocab.itos[topi.item()])

            decoder_input = topi.squeeze(0).detach()

        return decoded_words, decoder_attentions.squeeze(1)


def cal_bleu_score(data, model, source_vocab, target_vocab, device):
    model.eval()

    targets = []
    predictions = []

    for sample in data:
        src = sample.src
        trg = sample.trg
        predictions.append(
            translate(model, " ".join(src), source_vocab, target_vocab, device)[0])
        targets.append([trg])

    print(f'BLEU Score: {round(bleu_score(predictions, targets) * 100, 2)}')


def translate_base(model, sentence, src, trg, device, max_length=512):
    encoder, decoder = model.encoder, model.decoder
    sentence = sentence.lower()
    with torch.no_grad():
        input_tensor = [src.vocab.stoi[word] for word in sentence.split(' ')]
        input_tensor.append(src.vocab.stoi[src.eos_token])
        input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device).view(-1, 1)

        encoder_outputs, hidden = encoder(input_tensor)

        decoder_input = torch.tensor([trg.vocab.stoi[trg.init_token]], device=device)

        decoder_hidden = hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == trg.vocab.stoi[trg.eos_token]:
                break
            else:
                decoded_words.append(trg.vocab.itos[topi.item()])

            decoder_input = topi.squeeze(0).detach()

        return decoded_words


def cal_bleu_score_base(data, model, source_vocab, target_vocab, device):
    model.eval()

    targets = []
    predictions = []

    for sample in data:
        src = sample.src
        trg = sample.trg
        predictions.append(
            translate_base(model, " ".join(src), source_vocab, target_vocab, device))
        targets.append([trg])

    print(f'BLEU Score: {round(bleu_score(predictions, targets) * 100, 2)}')


def display_attention(sentence, model, SRC, TRG, device):
    r = translate(model, sentence, SRC, TRG, device)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(r[1][:len(r[0]) - 1, :-1], cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in sentence.split()] + [''],
                       rotation=45)
    ax.set_yticklabels([''] + r[0] + [''])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    print("src: ", sentence)
    print("trg: ", " ".join(r[0]))
    plt.show()
    plt.close()
