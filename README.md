# MIRGAN - Generating synthetic ncRNA data

MIRGAN is a Generative Adversarial Networks-based approach designed to create synthetic Non-coding RNAs (ncRNAs) data.

This repository also provides an alternative SMOTE-based approach to generating synthetic data.

# Instalation

Download the master folder and follow the steps below:

```
unzip MIRGAN-master.zip
```

Or git clone the MIRGAN respository:

```
git clone https://github.com/chiquitto/MIRGAN.git
```


# GAN

## The CONDA env

First of all, you must create the `mirgan` CONDA environment:

```bash
conda create python=3.10.11 --name mirgan -y
```

Activating the `mirgan` env:

```bash
conda activate mirgan
```

After that, you should install the depencies.

First, install the PyTorch depencies.
I advise following the instructions on the official page at https://pytorch.org/get-started/locally/.

And then install these:

```bash
conda install -y matplotlib
conda install -y -c conda-forge scikit-learn
```

## Preparing the input

The input should be a TXT file with only one nucleotide sequence per line.
See the `input/example_gan_input.csv` for further details.

Very important: The GAN should receive only one class of data.

Here is a example how to create the input file in the correct format, removing the unwanted class.
The shuffle command is optional.

```bash
awk 'BEGIN { FS = ","; OFS="," } { if ($3=="TRUE") { gsub("U","T",$2); print $2 } }' input/rna_sequences.csv | shuf > input/example_gan_input.txt
```

## Training the WGAN

```bash
python mirgan/mirgan.py --mode 0 --input input/example_gan_input.txt --outputdir output/mirgan_wgan_example
```

## Training the FBGAN

```bash
python mirgan/mirgan.py --mode 1 --input input/example_gan_input.txt --outputdir output/mirgan_fbgan_example
```

## Creating synthetic data

Using WGAN:

```bash
python mirgan/generate_samples.py --generator output/mirgan_wgan_example/models/final-generator.pt --samples 1000 --output output/example_wgan.txt
```

Using FBGAN:

```bash
python mirgan/generate_samples.py --generator output/mirgan_fbgan_example/models/final-generator.pt --samples 1000 --output output/example_fbgan.txt
```

## Training the CNN Analyzer component of FBGAN

The CNN Analyzer needs two CSV files as input, one file per class.
The CSV files needs two columns: id and seq.

```bash
echo "id,seq" > input/cnn_analyzer_mirtron.csv
echo "id,seq" > input/cnn_analyzer_mirna.csv

awk 'BEGIN { FS = ","; OFS="," } { if ($3=="TRUE") { gsub("U","T",$2); print $1, $2 } }' input/rna_sequences.csv | shuf >> input/cnn_analyzer_mirtron.csv
awk 'BEGIN { FS = ","; OFS="," } { if ($3=="FALSE") { gsub("U","T",$2); print $1, $2 } }' input/rna_sequences.csv | shuf >> input/cnn_analyzer_mirna.csv
```

```bash
python mirgan/analyzers/CnnAnalyzer/create_model.py
```

# SMOTE_DNA

Synthetic Minority Over-sampling TEchnique (SMOTE) for DNA, is a approach used to generate synthetic data based in nucleotide sequences.


## The CONDA env

First of all, you must create the `smote_dna` CONDA environment:

```bash
conda create python=3.10.11 --name smote_dna -y
```

Activating the `smote_dna` env:

```bash
conda activate smote_dna
```

And install the dependencies:

```bash
pip install imblearn
```

## Preparing the input

The input should be a CSV file.
The first column is the nucleotide sequence, and the second column is the class of the sequence.
See the `input/example_smote_input.csv` for further details.

Here is a example how to create the CSV file in the correct format.
The shuffle command is optional.

```bash
awk 'BEGIN { FS = ","; OFS="," } { gsub("U","T",$2); print $2, $3 }' input/rna_sequences.csv | shuf > input/example_smote_input.csv
```

## Creating synthetic data

To create synthetic data using the SMOTE_DNA, you can follow the example below.
Don't forget to activate the `smote_dna` environment.

```bash
python smote_dna/smote_dna.py --mode A --samples 1000 --input input/example_smote_input.csv --output output/example_smote_output.txt
```

Where:
* mode: use A, B or C. Read the work to understand the differences between the three modes;
* samples: number of samples created;
* input: path to csv file with real data used as input;
* output: path to output file containing the created samples.

This command will be create 1000 samples and save them in `output/example_smote_output.txt` file.
