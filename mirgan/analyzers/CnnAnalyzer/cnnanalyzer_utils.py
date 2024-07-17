import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

def gpu_test():
  device = None
  use_cuda = False
  if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
  else:
    device = torch.device("cpu")
  return (use_cuda, device)

def save_models(save_dir, model, model_id):
  save_dir = f'{save_dir}/{model_id}.pt'
  torch.save(model.state_dict(), save_dir)
  return save_dir

def eval_model(model, x_values, y_values, device):
    x_values = x_values.to(device=device)
    y_values = y_values.to(device=device)

    y_pred = model(x_values)
    y_pred = y_pred.reshape(-1)

    acc = (y_pred.round() == y_values).float().mean()
    
    return float(acc)

def read_input_csv(csv_file):
    with open(csv_file) as file_reader:
        csv_reader = csv.DictReader(file_reader)
        return [row['seq'] for row in csv_reader]

def contains_certain_characters(text, characters):
  return all(char in characters for char in text)
  
# def seq_encode(sequence, charmap):
#     one_hot = np.zeros((len(sequence), 5))
#     for i, nt in enumerate(sequence):
#         one_hot[i, charmap[nt]] = 1
#     return one_hot

def tokenizer(seq, charmap):
  return [ charmap[c] for c in seq ]

def onehot_tokenized(samples, onehot: OneHotEncoder):
  s = samples.shape
  return onehot.transform(samples.reshape(-1, 1)).toarray() \
    .reshape(s  + ( len(onehot.categories_[0]), ))
  # return torch.tensor(r, dtype=torch.float32)

def generate_onehot_encoder(charmap_len):
  table = np.arange(charmap_len).reshape(-1, 1)
  onehot = OneHotEncoder(dtype=np.float32)
  onehot.fit(table)
  return onehot
