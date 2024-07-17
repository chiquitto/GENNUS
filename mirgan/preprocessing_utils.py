from mirgan_utils import contains_certain_characters

def calc_charmap(seqs):
  # counts = collections.Counter(char for line in seqs for char in line)
  # for char, count in counts.most_common(2048-1):
  #   print(char, count)
  #   if char not in charmap:
  #     charmap[char] = len(inv_charmap)
  #     inv_charmap.append(char)

  charmap = {'P':0, 'A':1, 'T':2, 'G':3, 'C':4}
  inv_charmap = ['P', 'A', 'T', 'G', 'C']

  return charmap, inv_charmap

def seq_maxlenght(seq, max_length):
  if len(seq) > max_length:
    seq = seq[:max_length]
  else:
    # padding
    seq = seq + ( "P" * (max_length - len(seq)) )
  return seq

def filter_sequences(seqs, max_length):
  filtered_seqs = []
  allowed = ('A', 'T', 'G', 'C')
  for seq in seqs:
    seq = seq.upper().replace('U', 'T')
    
    if not contains_certain_characters(seq, allowed):
      continue

    filtered_seqs.append(list( seq_maxlenght(seq, max_length) ))
  return filtered_seqs

def tokenizer(seq, charmap):
  return [ charmap[c] for c in seq ]

def detokenizer(seq, inv_charmap):
  return [ inv_charmap[c] for c in seq ]