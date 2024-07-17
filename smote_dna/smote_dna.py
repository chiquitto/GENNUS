# %%
import csv
from datetime import datetime
import getopt
from imblearn.over_sampling import SMOTE
import numpy as np
import random
import sys

# %% [markdown]
# # Input args

# %%
def usage():
    # python smote_dna/smote_dna.py --mode A --charmap "N,A,C,G,T" --padid 0 --samples 1000 --input input.csv --output output.txt
    s = "USAGE: " \
        + "python smote_dna.py " \
        + "--mode [A,B,C] " \
        + "--samples int " \
        + "--input input.txt " \
        + "--output output.txt "
    print(s)
    
def process_argv(argv):

    requireds = ["mode", "samples", "input", "output"]
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
    r = { 'charmap': 'N,A,C,G,T', 'padid': 0 }
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
    argv = ['filename.py',
            "--mode", "A",
            # "--charmap", "N,A,C,G,T",
            # "--padid", "0",
            "--samples", "1000",
            "--input", "../input/example_smote_input.csv",
            "--output", "../tmp/smote_tmp.txt"
            ]
    opts = process_argv(argv)

if not is_notebook():
    opts = process_argv(sys.argv)

print("opts=", opts)

mode = opts['mode']
charmap = opts['charmap'].split(",")
padding_id = int(opts['padid'])
samples = int(opts["samples"])
file_input = opts['input']
file_output = opts['output']

padnt = charmap[padding_id]

# %% [markdown]
# # Opening the input file

# %%
# You need to open the file, and generate a list like:
# [ {seq:string, class:int} ]
# The seq value should be a string with the DNA sequence
# The class value should be a INTEGER that represents a class
# For example: miRNA=0 and mirtron=1
# Dont use strings to represent the class

# file_output = f"./generated/{run_id}.txt"

# %%
def seq_cleaning1(seq):
    return seq.upper().replace("U", "T")

# %%
# Loading a CSV file
sequences = []
with open(file_input, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    # csvreader = csv.DictReader(csvfile, fieldnames=["id","seq","class"])
    for row in csvreader:
        classint = 1 if row[1]=="TRUE" else 0
        seq = {"seq": seq_cleaning1(row[0]), "class": classint}

        sequences.append(seq)

# %% [markdown]
# # Preprocessing

# %%
charmap_inv = { v:k for k,v in enumerate(charmap) }

def seq_padding(seq, max_length):
    d = max_length - len(seq)
    if d > 0: return seq + ( "N" * d )
    else: return seq[:max_length]

def dna2int(seq):
    return [ charmap_inv[nt] for nt in seq ]

def seq_encode(seq):
    seq = seq_padding(seq, max_length)
    return dna2int(seq)

# %%
# Padding and convert sequences for integers

min_length = min( len(item["seq"]) for item in sequences )
max_length = max( len(item["seq"]) for item in sequences )

min_length0 = min( len(item["seq"]) for item in sequences if item["class"] == 0 )
max_length0 = max( len(item["seq"]) for item in sequences if item["class"] == 0 )

min_length1 = min( len(item["seq"]) for item in sequences if item["class"] == 1 )
max_length1 = max( len(item["seq"]) for item in sequences if item["class"] == 1 )

print(f"min_length={min_length}")
print(f"max_length={max_length}")
print(f"min_length0={min_length0}")
print(f"max_length0={max_length0}")
print(f"min_length1={min_length1}")
print(f"max_length1={max_length1}")

# %%
sequences_data = []
sequences_label = []
for item in sequences:
    seq = seq_encode(item["seq"])
    sequences_data.append(seq)
    sequences_label.append(item["class"])

# %% [markdown]
# # Obtaining X and Y

# %%
X = np.array(sequences_data)
Y = np.array(sequences_label)

# %% [markdown]
# # Generating data

# %%
def smote_filter1(data):
    has_padding = False

    # Test IF there is padding in middle of data
    for nt_number in data:
        if has_padding and nt_number != padding_id:
            return None
        elif nt_number == padding_id:
            has_padding = True
    return data

# %%
# Replace padding in middle for another nt
def smote_filter2(data):
    original_len = len(data)

    if isinstance(data, np.ndarray):
        data = data.tolist()

    # pop zeros from end
    while data[-1] == padding_id:
        data.pop()
    
    # replace padding for a random nt
    if padding_id in data:
        minimum, maximum = 1, 4
        for k in range(len(data)):
            if data[k] == padding_id:
                data[k] = random.randint(minimum, maximum)
    
    data_len = len(data)
    if data_len < original_len:
        data += [padding_id] * (original_len - data_len)

    return data

# %%
# Remove padding in middle
def smote_filter3(data):
    original_len = len(data)

    data = [ k for k in data if k != padding_id ]
    
    data_len = len(data)
    if data_len < original_len:
        data += [padding_id] * (original_len - data_len)

    return data

# %%
def smote_resample_filtered(X, Y,
    smote_filter_def,
    number_sampled = 0,
    class_upsampling = 1, smotting = 10**6):

    len_total = Y.shape[0]

    if number_sampled == 0:
        len_class1 = Y[ Y == class_upsampling ].shape[0]
        len_class0 = len_total - len_class1

        if len_class1 >= len_class0: return np.array([])
        number_sampled = len_class0 - len_class1

    X_return = []

    while smotting > 0:

        sm = SMOTE()
        X_sm, Y_sm = sm.fit_resample(X, Y)
        X_smoted = X_sm[ len_total: ]
        # Y_smoted = Y_sm[ len_total: ]

        # Find data without padding in middle
        for x_item in X_smoted:
            newX = smote_filter_def(x_item)
            if not newX is None:
                X_return.append(newX)

            if len(X_return) >= number_sampled:
                smotting = 0
                break

        smotting -= 1

    return np.array(X_return)

if mode == "A":
    smote_filter_def = smote_filter1
elif mode == "B":
    smote_filter_def = smote_filter2
elif mode == "C":
    smote_filter_def = smote_filter3
else:
    raise ValueError("MODE should be A, B or C")

newx = smote_resample_filtered(X, Y,
    smote_filter_def=smote_filter_def,
    number_sampled=samples )# , smotting=1)

newy = np.array([1] * newx.shape[0])

# %% [markdown]
# # Write to file

# %%
def seq_decode(seq):
    seq = "".join([ charmap[s] for s in seq ])
    return seq.rstrip("N")

new_sequences = []
for newx_item in newx:
    newx_item = seq_decode(newx_item)
    new_sequences.append(newx_item)

# %%
with open(file_output, 'w') as fp:
    for item in new_sequences:
        fp.write("%s\n" % item)

print("Output created: %s" % file_output)


