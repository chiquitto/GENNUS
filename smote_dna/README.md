# SMOTE_DNA

SMOTE_DNA is a Synthetic Minority Over-sampling TEchnique (SMOTE)-based adapted to generate synthetic nucleotide sequence data.

By following these instructions, you should be able to generate synthetic nucleotide sequences using SMOTE_DNA.
For more details on the different modes and algorithm-specific configurations, please refer to the original research publication.

# Table of Contents

- [Setting Up the Environment](#setting-up-the-environment)
- [Preparing the Input Data](#preparing-the-input-data)
- [Generating Synthetic Data](#generating-synthetic-data)

# Setting Up the Environment

First, create the `smote_dna` CONDA environment:

```bash
conda create python=3.10.11 --name smote_dna -y
```

Activate the `smote_dna` environment:

```bash
conda activate smote_dna
```

Then, install the required dependencies:

```bash
pip install imblearn
```

# Preparing the Input Data

The input should be a CSV file with two columns:
* The first column contains the nucleotide sequences.
* The second column contains the class labels for each sequence.

You can refer to the example file located at `input/example_smote_input.csv` for the expected format.

Here is an example command to create a CSV file in the required format from an existing dataset.
This command replaces all instances of "U" with "T" in the sequences.
The shuffle command is optional.

```bash
awk 'BEGIN { FS = ","; OFS="," } { gsub("U","T",$2); print $2, $3 }' input/rna_sequences.csv | shuf > input/example_smote_input.csv
```

# Generating Synthetic Data

To generate synthetic data using SMOTE_DNA, execute the following command.
Don't forget to activate the `smote_dna` environment.

```bash
python smote_dna/smote_dna.py --mode A --samples 1000 --input input/example_smote_input.csv --output output/example_smote_output.txt
```

Where:
* mode: choose between modes A, B, or C. Refer to the corresponding research paper to understand the differences between the modes;
* samples: the number of synthetic samples to generate;
* input: the path to the CSV file containing the real nucleotide sequence data (formatted as described above);
* output: the path where the generated synthetic data will be saved.

The command above will generate 1000 synthetic samples and save them in the `output/example_smote_output.txt` file.
