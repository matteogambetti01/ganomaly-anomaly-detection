# GANomaly — Anomaly Detection on MVTec Dataset

Unsupervised anomaly detection on the **MVTec bottle category** using the
[GANomaly architecture](https://arxiv.org/abs/1805.06725), implemented in PyTorch.

## Architecture
GANomaly is a GAN-based model that learns to reconstruct normal images.
At test time, anomalies are detected by measuring the reconstruction error
and the distance in latent space between encoder and decoder outputs.

## Dataset
[MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
— Bottle category. Training set contains only normal images; test set
contains both normal and anomalous samples.

## Results
| Metric    | Value |
|-----------|-------|
| AUROC     | 0.96  |
| PRAUC     | 0.99  |
| Precision | 0.94  |
| Recall    | 0.92  |
| F1        | 0.93  |

## Visualizations

**Anomaly score distribution** — red: anomalous, blue: normal

<img width="989" height="590" alt="download" src="https://github.com/user-attachments/assets/aa938079-6abd-4bab-8a21-3cc892652c72" />


**Latent space** (TensorBoard projector on test set)

<img width="733" height="490" alt="spazio-latente" src="https://github.com/user-attachments/assets/a2d2b737-13a9-4990-b2f8-7a5b752abefd" />


**Reconstruction example**

<img width="696" height="306" alt="ricostruzione-immagine" src="https://github.com/user-attachments/assets/d40b365b-24a4-4538-b486-8f775bda29e6" />


## Notebook Structure
1. Dataset loading — custom `Dataset` class for MVTec images
2. GANomaly network definition (Generator + Encoder + Discriminator)
3. Training loop with loss tracking via TensorBoard
4. Evaluation — anomaly scores, histogram, ROC curve
5. Latent space visualization

## ⚙️ Setup
```bash
pip install -r requirements.txt
# Open GANomaly_MVTec.ipynb in Jupyter or Colab
