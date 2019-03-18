from .model_interface import ModelI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import numpy as np


class LSTMCPM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, **kwargs):
        super(LSTMCPM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, **kwargs)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(self.embedding(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, _ = self.lstm(embedded_sents)
        out = self.output(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)

def batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i+batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]

def step(model, sents):
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y

def train_epoch(data, model, optimizer, batch_size):
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    for batch_idx, sents in enumerate(batches(data, batch_size)):
        model.zero_grad()
        _, loss, _ = step(model, sents)
        loss.backward()
        optimizer.step()

def evaluate(data, model, batch_size):
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, loss, y = step(model, sents)
            prob = out[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.neg().sum().item()
            word_count += y.data.shape[0]
        return loss, np.exp(entropy_sum / word_count)


class LSTMModel(ModelI):
    def __init__(self, hparams):
        self.vocab = hparams["vocab"]
        self.embedding_dim = hparams["embedding_dim"]
        self.hidden_dim = hparams["hidden_dim"]
        self.n_epochs = hparams["n_epochs"]
        self.batch_size = hparams["batch_size"]
        self.lr = hparams["lr"]
        self.dropout = hparams["dropout"]
        self.num_layers = hparams["num_layers"]
        self.model = LSTMCPM(len(self.vocab), self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

    def fit(self, train_dataset, val_dataset=None):
        train = []
        for sent in train_dataset:
            sent = self.vocab.encode_sentence(sent)
            train.append(sent)

        val = []
        if val_dataset is not None:
            for sent in val_dataset:
                sent = self.vocab.encode_sentence(sent)
                val.append(sent)


        for epoch_idx in range(self.n_epochs):
            logging.info("Training epoch {}".format(epoch_idx))
            train_epoch(train, self.model, self.optimizer, self.batch_size)
            train_loss, train_perplexity = evaluate(train, self.model, self.batch_size)
            logging.info("\tTrain loss: {:.3f}, perplexity: {:.2f}".format(train_loss, train_perplexity))
            if val:
                val_loss, val_perplexity = evaluate(val, self.model, self.batch_size)
                logging.info("\tValidation loss: {:.3f}, perplexity: {:.2f}".format(val_loss, val_perplexity))

    def log_score(self, progression):
        if not isinstance(progression, list):
            progression = progression.split(" ")
        progression = self.vocab.encode_sentence(progression)

        self.model.eval()
        with torch.no_grad():
            x = nn.utils.rnn.pack_sequence([torch.LongTensor(progression[:-1])])
            y = nn.utils.rnn.pack_sequence([torch.LongTensor(progression[1:])])

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            
            out = self.model(x)
            prob = out[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            log_score = prob.sum().item()
        return log_score