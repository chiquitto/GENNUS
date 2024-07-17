import torch.nn as nn

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

class BinaryClassificator(nn.Module):
  def __init__(self, n_chars, seq_len, hidden):
    self.version_id = '20240110_1852-0300'
    
    super(BinaryClassificator, self).__init__()
    self.n_chars = n_chars
    self.seq_len = seq_len
    self.hidden = hidden

    self.block = nn.Sequential(
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
      ResBlock(hidden),
    )
    self.conv1d = nn.Conv1d(n_chars, hidden, 1)

    # output layer
    self.linear = nn.Linear(seq_len * hidden, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):
    output = input.transpose(1, 2)
    output = self.conv1d(output)
    output = self.block(output)
    output = output.view(-1, self.seq_len * self.hidden)
    
    # output layer
    # output = self.linear(output)
    output = self.sigmoid(self.linear(output))
    return output
  

  def version(self):
    return "{instance}#{version}".format(instance=type(self).__name__, version=self.version_id)