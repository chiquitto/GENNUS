# %%
import csv
from datetime import datetime
import numpy as np
import sys
import torch

from architecture import Generator
from preprocessing_utils import detokenizer, calc_charmap
from mirgan_utils import gpu_test, to_var, decode_onehot

from generate_samples_args import process_argv, is_notebook

# %% [markdown]
# # Input args

# %%
if is_notebook():
    generator = "../tmp/test2/models/final-generator.pt"
    samples = 1000
    output = "../output/example_gan.txt"

    argv = ['filename.py',
            "--generator", generator,
            "--samples", samples,
            "--output", output
            ]
    opts = process_argv(argv)

if not is_notebook():
    opts = process_argv(sys.argv)

print("opts=", opts)

model_path = opts['generator']
number_samples = int(opts['samples'])
txt_output = opts['output']

# %%
remove_sequences_with_P = True

# %%
def generator_sample(batch_size):
  z_input = to_var(torch.randn(batch_size, 100))
  fake_samples = generator(z_input).detach()
  return fake_samples

def generate_sequences():
  r = []
  counter = 0
  while counter < number_samples:
    generated_samples = generator_sample(batch_size)
    generated_samples_np = (generated_samples.data).cpu().numpy()

    for seq in generated_samples_np:
      seq = decode_onehot(seq)
      seq = detokenizer(seq, inv_charmap)
      seq = ''.join(seq).rstrip('P')

      if remove_sequences_with_P and ('P' in seq):
          continue

      counter += 1
      r.append(seq)
  return r

# %%
def save_txt(path, output):
  if path is None: return
  
  with open(path, 'w') as txtfile:
    for seq in output:
      txtfile.write(seq + "\n")

# %%
use_cuda, device = gpu_test()
print(f'use_cuda={use_cuda} device={device}')

# %%
charmap, inv_charmap = calc_charmap('')

# %%
charmap_len = 5

max_side = 11
max_length = max_side * max_side

batch_size = 32

hidden = 512

generator = Generator( charmap_len, max_length, batch_size, hidden ).to(device=device)
generator.load_state_dict(torch.load(model_path))
generator.eval()

# %%
sequences = generate_sequences()

# %%
save_txt(txt_output, sequences)


