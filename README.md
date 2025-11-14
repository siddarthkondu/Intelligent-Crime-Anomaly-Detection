This project implements an image-based anomaly detection system for crime detection in surveillance frames.
A Convolutional Autoencoder (CAE) is trained on only normal surveillance images. During testing, frames that produce a high reconstruction error are flagged as anomalies, which may indicate suspicious or criminal activities.

This repository includes complete training, evaluation, inference, and visualization code.

📁 Project Structure
├── dataset.py        # Dataset loader for surveillance frames
├── models.py         # Convolutional Autoencoder model
├── train.py          # Model training script
├── evaluate.py       # Evaluation script (AUC, ROC, thresholds)
├── inference.py      # Inference on unseen data
├── utils.py          # Utility functions (saving, errors, plots)
├── visualize.py      # Visualization tools for reconstructions
├── data/
│   ├── train/        # Normal frames for training
│   ├── val/          # Optional validation frames
│   └── test/
│       ├── normal/   # Normal test frames
│       └── anomaly/  # Anomalous (crime) frames
└── checkpoints/      # Saved model weights

🚀 Features
✔️ Convolutional Autoencoder Model

Learns normal patterns in surveillance footage

Detects anomalies using reconstruction error

✔️ Complete Training Pipeline

MSE loss

TensorBoard logging

Automatic checkpoint saving

✔️ Evaluation Suite

Reconstruction error computation

ROC curve & AUC score

Threshold-based anomaly detection

Histogram comparison (normal vs anomaly errors)

✔️ Inference System

Runs prediction on any folder of frames

Outputs top anomalous frames with scores

✔️ Visualization Tools

View original vs reconstruction

Visualize anomalies

🔧 Installation
1. Clone the repository
git clone <your-repo-link>
cd intelligent-crime-anomaly-detection

2. Install dependencies
pip install -r requirements.txt

📊 Dataset Format

Your dataset must follow this structure:

data/
  train/        → normal frames only
  val/          → optional validation frames
  test/
    normal/     → normal test frames
    anomaly/    → anomalous test frames (crime-related)


Each folder should contain images:

frame01.jpg
frame02.png
...

🏋️ Training the Model

Run the training script:

python train.py \
  --data_root data \
  --img_size 128 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --latent_dim 128


Checkpoints are saved to:

checkpoints/best.pth
checkpoints/final.pth


TensorBoard logs will be available in:

runs/


View them via:

tensorboard --logdir runs

🧪 Evaluating the Model

Evaluation computes:

Reconstruction errors

AUC

ROC curve

Histograms

Error CSV file

TP/FP/TN/FN using threshold percentile

Run:

python evaluate.py \
  --data_root data \
  --checkpoint checkpoints/final.pth \
  --threshold_percentile 95


Outputs saved in:

eval_outputs/
  ├── error_hist.png
  ├── roc.png
  ├── errors.csv

🔍 Running Inference

To detect anomalies on any test set:

python inference.py \
  --data_root data \
  --ckpt checkpoints/final.pth \
  --img_size 224 \
  --batch_size 16


This prints:

AUC score

Top highest-error (most anomalous) frames

🧠 How It Works
1. Train on normal data only

The autoencoder learns to reconstruct normal frames.

2. Compute reconstruction error

Anomalous events (violence, robbery, accidents) cause high errors.

3. Thresholding

A percentile threshold (default: 95%) distinguishes anomaly vs normal.

📈 Example Outputs
🔹 ROC Curve

Saved as roc.png.

🔹 Error Histogram

Visualizes separation between normal & anomaly errors.

🔹 Reconstruction

visualize.py allows side-by-side comparison:

Original frame  →  Reconstructed frame

🛠️ Customization

You can modify:

latent_dim → controls bottleneck compression

img_size → resolution of frames

Autoencoder architecture in models.py

Threshold percentile in evaluate.py

🤝 Contributing

Feel free to submit PRs or improvements:

Use EfficientNet as encoder

Use ConvLSTM for video sequences

Add Streamlit/Flask UI

Improve anomaly score fusion
