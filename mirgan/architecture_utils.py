import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

def _sample_gumbel(shape, eps=1e-20, out=None):
  """
  Sample from Gumbel(0, 1)
  based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
  (MIT license)
  """
  U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
  return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, temp=1, eps=1e-20):
  """
  Draw a sample from the Gumbel-Softmax distribution
  based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
  (MIT license)
  """
  dims = logits.dim()
  gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
  y = logits + Variable(gumbel_noise)
  return F.softmax(y / temp, dims - 1)


def gumbel_softmax(logits, temp=1, hard=False, eps=1e-20):
  """
  Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temp: non-negative scalar temperature
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probability distribution that sums to 1 across classes

  Constraints: - this implementation only works on batch_size x num_features tensor for now
  based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
  (MIT license)
  """
  shape = logits.size()
  assert len(shape) == 2
  y_soft = _gumbel_softmax_sample(logits, temp=temp, eps=eps)
  if hard:
    _, k = y_soft.data.max(-1)
    # this bit is based on
    # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
    y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
    # this cool bit of code achieves two things:
    # - makes the output value exactly one-hot (since we add then
    #   subtract y_soft value)
    # - makes the gradient equal to y_soft gradient (since we strip
    #   all other gradients)
    y = Variable(y_hard - y_soft.data) + y_soft
  else:
    y = y_soft
  return y