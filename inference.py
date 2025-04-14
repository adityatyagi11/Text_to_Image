import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from train import VQVAE, TextToImageTransformer  
from train import  CONTEXT_LENGTH  
print("Helloooo12ooo1111o1ooo")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_image_from_text(text, vqvae, transformer, vocab):
    vqvae.eval()
    transformer.eval()
    
  
    tokens = [vocab.get(word, vocab["<unk>"]) for word in text.lower().split()]
    

    if len(tokens) > CONTEXT_LENGTH:
        tokens = tokens[:CONTEXT_LENGTH]
    else:
        tokens += [vocab["<pad>"]] * (CONTEXT_LENGTH - len(tokens))
    

    text_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad(), torch.amp.autocast(device_type= "cuda"):


        generated_indices = transformer(text_tensor, generate=True)

   
        generated_indices = generated_indices.view(1, 64, 64)  
        flat = generated_indices.view(-1)
        print("🔍 Generated token range:", flat.min().item(), "-", flat.max().item())
        print("🔍 Unique tokens:", torch.unique(flat))
        print("🧮 Token frequency:", torch.bincount(flat))


        generated_img = vqvae.decode(generated_indices)

   
        generated_img = (generated_img[0].cpu().numpy() + 1) / 2
        generated_img = np.transpose(generated_img, (1, 2, 0))  
        generated_img = generated_img.astype(np.float32)

    return generated_img



if __name__ == "__main__":
    
    from train import Flickr8kDataset

    dataset = Flickr8kDataset(
        img_dir="./New folder/Flicker8k_Dataset",
        token_file="./New folder/Flickr8k.token.txt",
        split_file="./New folder/Flickr_8k.trainImages.txt"
    )
    vocab = dataset.vocab

   
    vqvae = VQVAE()
    vqvae.load_state_dict(torch.load("vqvae_model.pth", map_location=device))
    vqvae.to(device)
    vqvae.eval()

    transformer = TextToImageTransformer(vocab_size=len(vocab))
    transformer.load_state_dict(torch.load("checkpoint_epoch_4.pt", map_location=device)['model_state_dict'])
    transformer.to(device)
    transformer.eval()

   
    test_prompts = [
        "a photo of a cat",
        "a landscape with mountains",
        "a city skyline",
        "an abstract painting"
    ]

   
    plt.figure(figsize=(12, 3))
    for i, prompt in enumerate(test_prompts):
        img = generate_image_from_text(prompt, vqvae, transformer, vocab)
        torch.cuda.empty_cache()
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.title(prompt, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.show()
