# Text-to-Image Generator (VQ-VAE + Transformer)

This project implements a custom **text-to-image generation model** from scratch using a two-stage architecture:
- A **Vector Quantized Variational Autoencoder (VQ-VAE)** to encode and decode images using discrete token representations
- A **Transformer decoder** that learns to generate image tokens conditioned on input text captions

Inspired by DALL·E-style models, this project demonstrates a fundamental understanding of vision-language modeling without relying on large-scale pre-trained APIs.

---

## 🔍 Highlights

- Trained a custom VQ-VAE on **8,000+ Flickr8k images**
- Built a 4-layer **Transformer decoder** to autoregressively predict **64×64 discrete image tokens** from captions
- Handled complete model training using **PyTorch** with manual gradient scaling and checkpointing
- Optimized GPU memory usage on **RTX 4060 (8GB)** using mixed precision to fit the entire training pipeline
- Designed an inference pipeline to generate images from raw text prompts using the trained models


---

## 🛠️ Tech Stack

- **Frameworks & Libraries:** PyTorch, NumPy, Matplotlib, Hugging Face Tokenizers
- **Modeling:** VQ-VAE, Autoregressive Transformer
- **Tools:** CUDA, Mixed Precision Training, tqdm
- **Dataset:** Flickr8k (images + captions)

---

## 🚀 How to Run

### 1. Clone the Repository


git clone https://github.com/adityatyagi11/Text_to_Image.git
cd Text_to_Image

2. Prepare the Dataset
Download the Flickr8k dataset and place the images and annotation files in the appropriate folders (see dataset loader structure in train.py).

3. Train the Models

python train.py
This will:

Train the VQ-VAE model first

Then train the Transformer to predict image tokens from captions

4. Run Inference

python inference.py
This will generate and save images based on sample prompts like:

"a photo of a cat"

"a landscape with mountains"

"a city skyline"

"an abstract painting"



📝 Note: Model output may appear grid-like initially and improves as the Transformer learns token diversity and alignment






