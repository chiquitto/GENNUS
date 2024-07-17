# %% [markdown]
# # Config

# %%
# Loading data config
POS_FILE = "./input/cnn_analyzer_mirtron.csv"
NEG_FILE = "./input/cnn_analyzer_mirna.csv"

OUTPUT_DIR = './output/cnnanalyzer_models'

# Preprocessing data config
MAXLEN = 121
PADNT = "N"
POS_LABEL = 1
NEG_LABEL = 0

# Model config
NUM_EPOCHS = 100
BATCH_SIZE = 512
LR = 0.0001
HIDDEN_LAYERS = 512

# %% [markdown]
# # Imports

# %%
import copy
import csv
from datetime import datetime
import json

import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from structure.BinaryClassificator import BinaryClassificator
from rna_dataset import RNADataset
from cnnanalyzer_utils import gpu_test, save_models, eval_model, \
        read_input_csv, contains_certain_characters, generate_onehot_encoder, \
        tokenizer, onehot_tokenized, eval_model

# %% [markdown]
# # gpu_test

# %%
use_cuda, device = gpu_test()
(use_cuda, device)

# %% [markdown]
# # runid

# %%
def gen_runid():
  return datetime.now().strftime('%Y%m%d_%H%M%S')
  
run_id = gen_runid()
run_dir = f"{OUTPUT_DIR}/{run_id}"

try:
    path = Path(run_dir)
    path.mkdir(parents=True)
    print(f'Created: {run_dir}')
except FileExistsError:
    pass

# %% [markdown]
# # def

# %%
# init values
acc_epochs = np.zeros(NUM_EPOCHS, dtype=np.float32)

# %%
def dump_config():
    config_obj =  {
        # Loading data config
        "pos_file": POS_FILE,
        "neg_file": NEG_FILE,

        # Preprocessing data config
        "maxlen": MAXLEN,
        "padnt": PADNT,
        "pos_label": POS_LABEL,
        "neg_label": NEG_LABEL,
        "onehot_charmap": onehot_charmap,
        "charmap": charmap,
        "inv_charmap": inv_charmap,
        "charmap_len": charmap_len,

        # Model config
        "model.version": classificator.version(),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "hidden_layers": HIDDEN_LAYERS
    }

    with open(f"{run_dir}/config.json", "w") as outfile:
        json.dump(config_obj, outfile, indent=2)


# %%
def dump_results():
    config_obj =  {
        "acc_epochs": acc_epochs.tolist(),
        "acc_max": float(acc_epochs.max())
    }
    # print(config_obj)

    with open(f"{run_dir}/results.json", "w") as outfile:
        json.dump(config_obj, outfile, indent=2)

# %%
def plot_metrics(acc):
    plt.plot(acc, label = 'Accuracy')

    plt.grid(which = "minor")
    plt.minorticks_on()

    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(10, 5)
    # plt.show()
    plt.savefig(f"{run_dir}/metrics.png", bbox_inches='tight', dpi=200)
    plt.close()

# %% [markdown]
# # Load RNA Sequences

# %%
def filter_sequences(seqs):
  filtered_seqs = []
  allowed = ('A', 'T', 'G', 'C')
  for seq in seqs:
    seq = seq.upper().replace('U', 'T')
    
    if not contains_certain_characters(seq, allowed):
      continue

    filtered_seqs.append(seq.ljust(MAXLEN, PADNT)[:MAXLEN])
  return filtered_seqs
  
posdata = filter_sequences(read_input_csv(POS_FILE))
negdata = filter_sequences(read_input_csv(NEG_FILE))
data_sequences = posdata + negdata
labels = (POS_LABEL,) * len(posdata) + (NEG_LABEL,) * len(negdata)
# data_sequences = [ (d, POS_LABEL) for d in posdata ] + [ (d, NEG_LABEL) for d in negdata ]

# %% [markdown]
# # pre-process sequences

# %%
# Define a function to convert RNA sequences to one-hot encoding
onehot_charmap = {'N':0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

# %%
def calc_charmap(loaded_):
  charmap = {'N':0, 'A':1, 'T':2, 'G':3, 'C':4}
  inv_charmap = ['N', 'A', 'T', 'G', 'C']
  return charmap, inv_charmap

charmap, inv_charmap = calc_charmap(data_sequences)
charmap_len = len(charmap)

# %% [markdown]
# # Prepare data to model

# %%
def prepare_data_model(data, labels):
  tokenized_seqs = [ tokenizer(seq, charmap) for seq in data ]

  prepared_data = onehot_tokenized(np.array(tokenized_seqs, dtype=np.float32), onehot)
  prepared_labels = np.array(labels, dtype=np.float32)

  return prepared_data, prepared_labels

onehot = generate_onehot_encoder(charmap_len)
data_sequences_prepared, labels_prepared = prepare_data_model(data_sequences, labels)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    data_sequences_prepared, labels_prepared,
    test_size=0.2, stratify=labels_prepared)

# %%
train_dataset = RNADataset(X_train, Y_train)
test_dataset = RNADataset(X_test, Y_test)

# %%
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# # Instance classificator

# %%
classificator = BinaryClassificator( len(onehot_charmap), MAXLEN, HIDDEN_LAYERS ).to(device=device)

# %% [markdown]
# # Train

# %%
best_acc = 0
best_weights = None

# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.BCELoss()
optimizer_classificator = optim.Adam(classificator.parameters(), lr=LR)

print(f"Using CUDA: {use_cuda}, Device: {device}")

for epoch in range(NUM_EPOCHS):
    epoch_user = epoch + 1
    classificator.train()

    for x_real, y_real in trainloader:

        # classificator.zero_grad()

        x_real = x_real.to(device=device)
        y_real = y_real.to(device=device)

        # foward pass
        y_pred = classificator(x_real)
        y_pred = y_pred.reshape(-1)

        # loss
        loss = loss_fn(y_pred, y_real)

        # backward pass
        optimizer_classificator.zero_grad()
        loss.backward()

        # update weights
        optimizer_classificator.step()

    classificator.eval()
    # for x_test, y_test in testloader:
    x_test, y_test = next(iter(testloader))
    acc = eval_model(classificator, x_test, y_test, device)
    acc_epochs[epoch] = acc

    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(classificator.state_dict())

    if epoch_user % 10 == 0:
        print("==> Epoch {epoch}/{maxepoch} - acc={acc:.3f}".format(epoch=epoch_user,maxepoch=NUM_EPOCHS,acc=acc))
        plot_metrics(acc_epochs)

# save the final model
r = save_models(run_dir, classificator, "last")
print("Last model saved to: {path} (acc:{acc:.3f})".format(path=r,acc=acc_epochs[-1]))

classificator.load_state_dict(best_weights)
r = save_models(run_dir, classificator, "best")
print("Best model saved to: {path} (acc:{acc:.3f})".format(path=r,acc=best_acc))


# %% [markdown]
# # Saving config

# %%
dump_config()
dump_results()


