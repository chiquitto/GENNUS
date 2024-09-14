# MIRGAN - Generating synthetic ncRNA data with GAN

MIRGAN is a Generative Adversarial Networks-based approach designed to create synthetic Non-coding RNAs (ncRNAs) nucleotide sequences.

# Table of Contents
- [Setting Up the Environment](#setting-up-the-environment)
- [Preparing the Input Data](#preparing-the-input-data)
- [Training Models](#training-models)
  - [Training the WGAN](#training-the-wgan)
  - [Training the FBGAN](#training-the-fbgan)
- [Generating Synthetic Data](#generating-synthetic-data)
- [Training the CNN Analyzer for FBGAN](#training-the-cnn-analyzer-for-fbgan)
- [Building a Custom Analyzer for FBGAN](#building-a-custom-analyzer-for-fbgan)


# Setting Up the Environment

First, create the `mirgan` CONDA environment:

```bash
conda create python=3.10.11 --name mirgan -y
```

Activate the `mirgan` environment:

```bash
conda activate mirgan
```

Next, install the necessary dependencies.

Begin by installing PyTorch. Follow the official installation instructions at https://pytorch.org/get-started/locally/.

Then, install the remaining dependencies:

```bash
conda install -y matplotlib
conda install -y -c conda-forge scikit-learn
```

# Preparing the Input Data

The input for MIRGAN should be a text file (TXT) with one nucleotide sequence per line.
You can refer to the example file at `input/example_gan_input.csv`.

Very important: The GAN model should receive data from only one class at a time.

Below is an example of how to generate the input file in the required format from a CSV file, ensuring that only one class is included.
This example removes unwanted classes and outputs one sequence per line.
The shuf command is optional but recommended to shuffle the sequences.

```bash
awk 'BEGIN { FS = ","; OFS="," } { if ($3=="TRUE") { gsub("U","T",$2); print $2 } }' input/rna_sequences.csv | shuf > input/example_gan_input.txt
```

# Training Models

The datasets and models generated during the work are available for download from <http://zenodo.com/10.5281/zenodo.13235408>.
You can use these datasets to train GANs, or use the pre-trained models.

## Training the WGAN

To train the WGAN model using the prepared input file, run the following command:

```bash
python mirgan/mirgan.py --mode 0 --input input/example_gan_input.txt --outputdir output/mirgan_wgan_example
```

## Training the FBGAN

To train the FBGAN model, you need to configure an Analyzer Component.

This package includes a pre-configured CNN Analyzer Component, which uses a pre-trained model.
You should download the pre-trained model from <http://zenodo.com/10.5281/zenodo.13235408>.

Once downloaded, set the `CNNANALYZER_MODEL_PATH` parameter in `mirgan/analyzer_loader.py` to the path of the pre-trained model.

Once the analyzer is configured, you can train the FBGAN model with the following command:

```bash
python mirgan/mirgan.py --mode 1 --input input/example_gan_input.txt --outputdir output/mirgan_fbgan_example
```

Alternatively, you can:

* Train the CNN Analyzer with your own data (section `Training the CNN Analyzer for FBGAN`);
* Build your own analyzer (section `Building a Analyzer component for FBGAN`).

# Generating Synthetic Data

Once the models are trained, you can generate synthetic data using either WGAN or FBGAN:

Using WGAN:

```bash
python mirgan/generate_samples.py --generator output/mirgan_wgan_example/models/generator.pt --samples 1000 --output output/example_wgan.txt
```

Using FBGAN:

```bash
python mirgan/generate_samples.py --generator output/mirgan_fbgan_example/models/generator.pt --samples 1000 --output output/example_fbgan.txt
```

# Training the CNN Analyzer for FBGAN

To train the CNN Analyzer component, you will need two CSV files—one for each class.

**Preparing the Input Datasets**

The CNN Analyzer is a binary classificator.

Training requires two CSV files, each with samples from only one class.
Each CSV file should contain two columns: id and seq.

Below is an example of how to create these CSV files from an existing dataset:

```bash
echo "id,seq" > input/cnn_analyzer_mirtron.csv
echo "id,seq" > input/cnn_analyzer_mirna.csv

awk 'BEGIN { FS = ","; OFS="," } { if ($3=="TRUE") { gsub("U","T",$2); print $1, $2 } }' input/rna_sequences.csv | shuf >> input/cnn_analyzer_mirtron.csv
awk 'BEGIN { FS = ","; OFS="," } { if ($3=="FALSE") { gsub("U","T",$2); print $1, $2 } }' input/rna_sequences.csv | shuf >> input/cnn_analyzer_mirna.csv
```

**Training the CNN Analyzer**

Once the input datasets are ready, you can train the CNN Analyzer with the following command:

```bash
python mirgan/analyzers/CnnAnalyzer/create_model.py
```

The trained model will be saved in `./output/cnnanalyzer_models`.
After training, update the `CNNANALYZER_MODEL_PATH` parameter to the path of your newly trained model.

# Building a Custom Analyzer for FBGAN

You can create your own analyzer by implementing a scoring mechanism for synthetic samples. The analyzer should assign a score between 0 and 1, where 1 indicates a highly desirable sample.

To build a custom analyzer:

1. Create a new analyzer class that inherits from `AnalyzerInterface`. Use the CNNAnalyzer as example (`mirgan/analyzers/CnnAnalyzer/cnn_analyzer.py`).
2. Modify `mirgan/analyzer_loader.py` to load your custom Analyzer Component.
