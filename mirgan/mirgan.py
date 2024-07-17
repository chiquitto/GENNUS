# %% [markdown]
# # Libraries & definitions

# %%
import sys
 
# setting path
sys.path.append('.')
sys.path.append('../')

# %%
import csv
from datetime import datetime
import getopt

import numpy as np
import os
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from mirgan_utils import gpu_test, load_dataset, create_ones, generate_onehot_encoder, \
  to_var, torch_onehot, decode_onehot, seq_data_exploration, save_models
from preprocessing_utils import calc_charmap, filter_sequences, tokenizer, detokenizer, seq_maxlenght
from architecture import Discriminator, Generator

# %% [markdown]
# # Input args

# %%
def usage():
    # mode = 0 | WGAN
    # mode = 1 | FBGAN + CnnAnalyzer

    # python mirgan/mirgan.py --mode 0 --input input.txt --outputdir directory
    s = "USAGE: " \
        + "python mirgan/mirgan.py " \
        + "--mode int " \
        + "--input input.txt " \
        + "--outputdir directory "
    print(s)
    
def process_argv(argv):

    requireds = ["mode", "input", "outputdir"]
    input_args = requireds + ['help']

    try:
        longopts = [ opt + "=" for opt in input_args ]
        opts, args = getopt.getopt(argv[1:], "", longopts)
    except getopt.GetoptError as e:
        print("Wrong usage!")
        print(e)
        usage()
        sys.exit(1)

    # parse the options
    r = {}
    for op, value in opts:
        op = op.replace('--', '')
        if op == 'help':
            usage()
            sys.exit()
        elif op in input_args:
            r[op] = value

    for required in requireds:
        if not required in r:
            print("Wrong usage!!")
            print("Param {} is required".format(required))
            usage()
            sys.exit(1)

    return r

def is_notebook() -> bool:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# %%
if is_notebook():
    mode = "0"
    input = '../input/example_gan_input.txt'
    outputdir = '../tmp/test1'

    argv = ['filename.py',
            "--mode", mode,
            "--input", input,
            "--outputdir", outputdir
            ]
    opts = process_argv(argv)

if not is_notebook():
    opts = process_argv(sys.argv)

print("opts=", opts)

mode = int(opts['mode'])
if mode < 0 or mode > 1:
    print("Mode arg should be 0 or 1.")
    usage()
    raise

file_input = opts['input']
output_dir = opts['outputdir']

# %%
# Definitions

# torch.manual_seed(111)

max_side = 11
max_length = max_side * max_side

batch_size = 32
retain_generated_sequences = 200

lr = 0.0001
num_epochs = 100
num_epochs_discriminator = 5
num_epochs_generator = 1
hidden = 512

lamda = 10 #lambda

models_dir = f'{output_dir}/models'
generated_dir = f'{output_dir}/generated'
tmp_dir = f'{output_dir}/tmp'

# %%
print(f"models_dir={models_dir}")
print(f"generated_dir={generated_dir}")
print(f"tmp_dir={tmp_dir}")

# %%
try:
    path = Path(output_dir)
    path.mkdir(parents=True)
except FileExistsError:
    # print(f"FileExistsError: {output_dir} already exists")
    pass

try:
    path = Path(models_dir)
    path.mkdir(parents=True)
except FileExistsError:
    pass

try:
    path = Path(generated_dir)
    path.mkdir(parents=True)
except FileExistsError:
    pass

try:
    path = Path(tmp_dir)
    path.mkdir(parents=True)
except FileExistsError:
    pass

# %% [markdown]
# # GPU test

# %%
use_cuda, device = gpu_test()
(use_cuda, device)

# %% [markdown]
# # Open dataset

# %%
# Open dataset and return list of sequences
loaded_ds = load_dataset(file_input, limit=None)
loaded_length = len(loaded_ds)

# %% [markdown]
# # Preprocessing

# %%
charmap, inv_charmap = calc_charmap(loaded_ds)
filtered_seqs = filter_sequences(loaded_ds, max_length=max_length)
charmap_len = len(charmap)

# %% [markdown]
# # Preparing the Training Data

# %%
def seqs_to_trainset(input_seqs):
  tokenized_seqs = [ tokenizer(seq, charmap) for seq in input_seqs ]

  train_data_length = len(tokenized_seqs)
  train_data = torch.tensor(np.array(tokenized_seqs))

  train_labels = torch.zeros(train_data_length)

  r = [ (x,y) for (x,y) in zip(train_data, train_labels) ]
  return r

train_set = seqs_to_trainset(filtered_seqs)

# %% [markdown]
# # Discriminator and the Generator

# %%
discriminator = Discriminator( charmap_len, max_length, batch_size, hidden ).to(device=device)
generator = Generator( charmap_len, max_length, batch_size, hidden ).to(device=device)

# %% [markdown]
# # Training utils

# %%
# Adam algorithm to train the discriminator and generator models
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))

# %%
onehot = generate_onehot_encoder(charmap_len)

# %%
def calc_gradient_penalty(real_data, fake_data):
  alpha = torch.rand(batch_size, 1, 1)
  alpha = alpha.view(-1,1,1)
  alpha = alpha.expand_as(real_data)
  alpha = alpha.cuda() if use_cuda else alpha
  interpolates = alpha * real_data + ((1 - alpha) * fake_data)

  interpolates = interpolates.cuda() if use_cuda else interpolates
  interpolates = autograd.Variable(interpolates, requires_grad=True)

  disc_interpolates = discriminator(interpolates)

  gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                if use_cuda else torch.ones(disc_interpolates.size()),
                create_graph=True, retain_graph=True, only_inputs=True)[0]

  gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * lamda
  return gradient_penalty

# %%
one, one_neg = create_ones(use_cuda)

# %%
def generator_sample(batch_size):
  z_input = to_var(torch.randn(batch_size, 100))
  fake_samples = generator(z_input).detach()
  return fake_samples

# %%
def get_train_loader(dataset):
  return torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
  )

# %% [markdown]
# # Analyzer

# %%
# mode = 0 | WGAN
# mode = 1 | FBGAN + CnnAnalyzer

analyzer = None
analyzer_threshold = 0

if mode == 1:
    from analyzers.CnnAnalyzer.cnn_analyzer import CnnAnalyzer

    analyzer = CnnAnalyzer(loaded_ds,
        model_path="mirgan/analyzers/CnnAnalyzer/cnnanalyzer_models/mirtron_mirna/best.pt",
        device=device)
    analyzer_threshold = 0.95

# %%
def save_sequences(sequences, save_dir, prefix):
  output_file = f'{save_dir}/{prefix}.txt'
  with open(output_file, 'w') as fp:
    for seq in sequences:
      fp.write("%s\n" % seq)

def save_trainset(sequences, save_dir, prefix):
  output = []
  for seq in sequences:
    seq = detokenizer(seq[0], inv_charmap)
    output.append( ''.join(seq).rstrip('P') )
  save_sequences( output, save_dir, f"trainset_{prefix}" )

def generated_sample_to_string(generated_sequence):
  seq = ''.join(detokenizer(decode_onehot(generated_sequence), inv_charmap))
  seq = seq.rstrip('P')

  return seq

def analyzer_filter2(generated_sequence):
  if 'P' in generated_sequence: return None
  if len(generated_sequence) < 50: return None
  return generated_sequence

def analyzer_run(epoch):
  if analyzer is None: return []
  
  generated_samples = generator_sample(batch_size)
  generated_samples_np = (generated_samples.data).cpu().numpy()

  generated_samples_filtered = []
  generated_samples_tosave = []
  for n, seq in enumerate(generated_samples_np):
    seq1 = generated_sample_to_string(seq)
    generated_samples_tosave.append( seq1 )

    seq2 = analyzer_filter2(seq1)

    if not seq2 is None:
      generated_samples_filtered.append( { 'id':'seq%04d' % n, 'seq': seq2 } )

  save_sequences(generated_samples_tosave, generated_dir, "generated_samples_%04d" % epoch)

  if not generated_samples_filtered: return []

  # run analyzer
  analyzer.setInput(generated_samples_filtered)
  analyzer.prepare()
  analyzer.run()
  analyzer_results = analyzer.getScores()
  # analyzer.clear()

  # filter analyzer results
  positive_sequences = []
  scored_sequences = []
  for generated_sequence in generated_samples_filtered:

    scored_sequences.append( {"id": generated_sequence['id'], \
      "score": round(analyzer_results[ generated_sequence['id'] ]['score'], 3), \
      "seq": generated_sequence["seq"]} )

    if analyzer_results[ generated_sequence['id'] ]['score'] >= analyzer_threshold:
      positive_sequences.append( tuple( seq_maxlenght(generated_sequence['seq'], max_length) ) )
  
  if scored_sequences:
    with open(f'{generated_dir}/scores_{epoch}.csv', 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, scored_sequences[0].keys())
      dict_writer.writeheader()
      dict_writer.writerows(scored_sequences)

  return positive_sequences

# %% [markdown]
# # Training

# %%
def train_discriminator(real_samples):
  real_data = torch_onehot(real_samples, onehot).to(device=device)
  
  discriminator.zero_grad()
  d_real_pred = discriminator(real_data)
  d_real_err = torch.mean(d_real_pred) #want to push d_real as high as possible
  d_real_err.backward(one_neg)

  # d_fake_data = generator_sample(batch_size)
  z_input = to_var(torch.randn(batch_size, 100))
  d_fake_data = generator(z_input).detach()
  
  d_fake_pred = discriminator(d_fake_data)
  d_fake_err = torch.mean(d_fake_pred) #want to push d_fake as low as possible
  d_fake_err.backward(one)

  gradient_penalty = calc_gradient_penalty(real_data.data, d_fake_data.data)
  gradient_penalty.backward()

  d_err = d_fake_err - d_real_err + gradient_penalty
  optimizer_discriminator.step()

  return d_err

# %%
def train_generator():
  generator.zero_grad()

  # g_fake_data = generator_sample(batch_size)
  z_input = to_var(torch.randn(batch_size, 100))
  g_fake_data = generator(z_input)
  
  dg_fake_pred = discriminator(g_fake_data)
  g_err = -torch.mean(dg_fake_pred)
  g_err.backward()
  optimizer_generator.step()

  return g_err, g_fake_data

# %%
def save_loss(loss_discriminator, loss_generator, analyzer_positives, mse_list):
    plt.plot(loss_discriminator, label = 'D_loss')
    plt.plot(loss_generator, label = 'G_loss')
    plt.plot(analyzer_positives, label = 'Analyzer Positives')
    plt.plot(mse_list, label = 'MSE')

    loss_sum = np.abs(loss_discriminator) + np.absolute(loss_generator)
    plt.plot(loss_sum, label = 'Sum')

    plt.grid(which = "minor")
    plt.minorticks_on()

    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(10, 5)
    # plt.show()
    plt.savefig(f"{output_dir}/chart.png", bbox_inches='tight', dpi=200)
    plt.close()

    with open(f"{output_dir}/chart.json", 'w') as fp:
        d = {
            "loss_discriminator": loss_discriminator.tolist(),
            "loss_generator": loss_generator.tolist(),
            "analyzer_positives": analyzer_positives.tolist(),
            "mse_list": mse_list.tolist(),
        }
        json.dump(d, fp)

# %%
def calc_mse_seqs(A, B):
    mse_sum = 0
    for k, a in enumerate(A):
        mse_min = min([ calc_mse_sequence(a, b) for b in B ])
        mse_sum += mse_min
    return mse_sum

def calc_mse_sequence(seq_a, seq_b):
    mse_sum = 0
    for nt_a, nt_b in zip(seq_a, seq_b):
        if np.argmax(nt_a) == np.argmax(nt_b):
            continue
        mse_sum += calc_mse_nt(nt_a, nt_b)
    return mse_sum / seq_a.shape[0]

def calc_mse_nt(nt_a, nt_b):
    return np.square(np.subtract(nt_a, nt_b)).mean()

mse_real_data = np.array( [train_set_item[0] for train_set_item in train_set] )
mse_real_data = torch_onehot(mse_real_data, onehot)

# %%
loss_discriminator_list = np.zeros(num_epochs)
loss_generator_list = np.zeros(num_epochs)
analyzer_positives_list = np.zeros(num_epochs)
mse_list = np.zeros(num_epochs)

def train_gan():
  print("device=", device)

  generated_positives = []

  for epoch in range(1, num_epochs + 1):
    print(f"==> Epoch: {epoch}")

    # train_set.extend( seqs_to_trainset(positive_sequences) )
    analyzer_positives = analyzer_run(epoch)
    analyzer_positives_len = len(analyzer_positives) if analyzer_positives else 0
    analyzer_positives_list[epoch-1] = analyzer_positives_len
    if analyzer_positives:
      generated_positives.extend(analyzer_positives)
      generated_positives = generated_positives[-retain_generated_sequences:]
      print(f"Analyzer: added {analyzer_positives_len} seqs")
    
    train_set_positives = seqs_to_trainset(generated_positives)
    train_loader = get_train_loader(train_set + train_set_positives)

    if epoch % 10 == 0:
      save_trainset(train_set_positives, generated_dir, "epoch_%04d" % epoch)

    d_err, g_err, g_fake_data = None, None, []

    for n, (real_samples, real_labels) in enumerate(train_loader):
      
      # Train Discriminator
      for i in range(num_epochs_discriminator):
        d_err = train_discriminator(real_samples)

      # Train Generator
      for i in range(num_epochs_generator):
        g_err, g_fake_data = train_generator()

    # Show loss
    if not d_err is None:
      loss_discriminator = (d_err.data).cpu().numpy()
      loss_generator = (g_err.data).cpu().numpy()

      loss_discriminator_list[epoch-1] = loss_discriminator
      loss_generator_list[epoch-1] = loss_generator

      g_fake_data = generator_sample(32).cpu().numpy()
      mse = calc_mse_seqs(g_fake_data, mse_real_data)
      mse_list[epoch-1] = mse
      print(f"MSE: {mse}")

      print(f"Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
      save_loss(loss_discriminator_list, loss_generator_list, analyzer_positives_list, mse_list)

      # save_models(models_dir, discriminator, generator, str(epoch))

train_gan()

# %%
save_models(models_dir, discriminator, generator, "final")


