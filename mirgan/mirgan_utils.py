import numpy as np

from sklearn.preprocessing import OneHotEncoder

import torch
from torch.autograd import Variable

def gpu_test():
  device = None
  use_cuda = False
  if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
  else:
    device = torch.device("cpu")
  return (use_cuda, device)

def load_dataset(file, limit=None):
  with open(file, 'r', encoding='utf8') as f:
    lines = [ line.strip() for line in f.readlines()]

    if limit is None:
      return lines
    else: return lines[:limit]

def contains_certain_characters(text, characters):
  return all(char in characters for char in text)

def generate_onehot_encoder(charmap_len):
  table = np.arange(charmap_len).reshape(-1, 1)
  onehot = OneHotEncoder()
  onehot.fit(table)
  return onehot

def torch_onehot(samples, onehot: OneHotEncoder):
  s = samples.shape
  r = onehot.transform(samples.reshape(-1, 1)).toarray() \
    .reshape(s  + ( len(onehot.categories_[0]), ))
  return torch.tensor(r, dtype=torch.float32)

# input = numpy.ndarray (100, 5)
def decode_onehot(onehot_encoded):
  r = []
  for row in onehot_encoded:
    r.append( np.argmax(row) )
  return r

# _real_samples
# def samples_to_device(samples, device):
#   samples = torch_onehot(samples)
#   samples = samples.to(device=device)
#   return samples

def create_ones(use_cuda=None):
  # one = torch.FloatTensor([1])
  one = torch.tensor(1, dtype=torch.float)
  if use_cuda is None: use_cuda = torch.cuda.is_available()
  one = one.cuda() if use_cuda else one
  one_neg = one * -1
  return one, one_neg

# input = decode_onehot return
def decode_seq(seq_encoded, inv_charmap):
  return [ inv_charmap[nt] for nt in seq_encoded ]

def to_var(x):
  if torch.cuda.is_available():
    x = x.cuda()
  # return Variable(x)
  return x

def seq_data_exploration(seqs, letters):
  counts = { key:0 for key in letters }
  counts['total'] = 0

  for seq in seqs:
    for nt in letters:
      c = seq.count(nt)
      counts[nt] += c
      counts['total'] += c

  return (counts, { k : round(v / counts['total'], 2) for k,v in counts.items() })

def save_models(save_dir, discriminator_model, generator_model, prefix):
  torch.save(discriminator_model.state_dict(), f'{save_dir}/{prefix}-discriminator.pt')
  torch.save(generator_model.state_dict(), f'{save_dir}/{prefix}-generator.pt')
