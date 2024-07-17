from torch import nn

from architecture_utils import gumbel_softmax

class ResBlock(nn.Module):
  def __init__(self, hidden):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(
      nn.ReLU(True),
      nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
      nn.ReLU(True),
      nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
    )

  def forward(self, input):
    output = self.res_block(input)
    return input + (0.3*output)

class Discriminator(nn.Module):
  def __init__(self, n_chars, seq_len, batch_size, hidden):
    super(Discriminator, self).__init__()
    self.n_chars = n_chars
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.hidden = hidden

    self.block = nn.Sequential(
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
    )
    self.conv1d = nn.Conv1d(n_chars, hidden, 1)
    self.linear = nn.Linear(seq_len * hidden, 1)

  def forward(self, input):
    output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
    output = self.conv1d(output)
    output = self.block(output)
    output = output.view(-1, self.seq_len * self.hidden)
    output = self.linear(output)
    return output

class Generator(nn.Module):
  # n_chars=5 max_length=225 batch_size=64 hidden=512
  def __init__(self, n_chars, max_length, batch_size, hidden):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(100, hidden * max_length)
    self.block = nn.Sequential(
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
    )
    self.conv1 = nn.Conv1d(hidden, n_chars, 1)

    self.n_chars = n_chars
    self.max_length = max_length
    self.batch_size = batch_size
    self.hidden = hidden

  def forward(self, noise):
    output = self.fc1(noise)
    output = output.view(-1, self.hidden, self.max_length) # (BATCH_SIZE, DIM, max_length)
    output = self.block(output)
    output = self.conv1(output)
    output = output.transpose(1, 2)
    shape = output.size()
    output = output.contiguous()
    output = output.view(self.batch_size * self.max_length, -1)
    output = gumbel_softmax(output, 0.5)
    return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))