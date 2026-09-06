# NSSC HiRISE V2 - CNN Autoencoder for Anomaly Detection

A convolutional autoencoder built with TensorFlow/Keras to detect anomalies in Mars HiRISE (High Resolution Imaging Science Experiment) satellite imagery from the NSSC HiRISE dataset. The model learns to reconstruct normal surface images, then flags anomalies using a combination of reconstruction error, structural similarity (SSIM), and an Isolation Forest applied to the learned latent space.

## Overview

The pipeline:

1. Loads and preprocesses grayscale HiRISE images (227x227).
2. Trains a convolutional autoencoder (encoder-decoder) with a 256-dimensional latent space.
3. Uses a combined MSE + SSIM loss for reconstruction.
4. Computes per-image reconstruction error and SSIM on the test set.
5. Trains an Isolation Forest on the latent representations for novelty scoring.
6. Combines both signals to rank and report the top suspected anomalies.
7. Saves results (CSV), latent vectors (NPY), and the trained model.

## Model Architecture

- Encoder: 5 convolutional blocks (32 to 256 filters) with batch normalization, followed by global average pooling and a dense latent layer.
- Decoder: Dense projection followed by 4 transposed convolution blocks with batch normalization, a bilinear resize, and a final sigmoid convolution for reconstruction.
- Loss: Weighted combination of pixel-wise MSE (0.7) and SSIM-based structural loss (0.3).

## Dataset

This project expects the NSSC HiRISE dataset, structured with `train`, `validation`, and `test` splits (each containing a `normal` subfolder of images) along with `metadata.csv` files and a `source_group_mapping.csv` for grouping related images.

The dataset itself is not included in this repository. Place your dataset zip (for example `NSSC_HiRISE.zip`) in the working directory before running the script, and update the paths in the script if needed.

## Requirements

See `requirements.txt`. Key dependencies:

- TensorFlow / Keras
- NumPy
- Pandas
- Pillow
- Matplotlib
- scikit-learn

## Usage

This script was originally developed in Google Colab and includes Colab-specific calls (`google.colab.files.upload`, `files.download`). To run it locally:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Remove or replace the Colab-specific upload/download calls with local file paths.
3. Place the dataset zip in the expected location and update `ZIP_PATH` / `EXTRACT_DIR` as needed.
4. Run the script:
   ```
   python v2.py
   ```

## Output

Running the full pipeline produces:

- `NSSC_HiRISE_V2.keras` - trained autoencoder model
- `NSSC_HiRISE_V2_results.csv` - per-image anomaly scores and metadata
- `NSSC_HiRISE_V2_Z_train.npy`, `NSSC_HiRISE_V2_Z_test.npy` - latent representations
- A backup zip containing the above artifacts

## License

This project is licensed under the MIT License. See `LICENSE` for details.
