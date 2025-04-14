import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from PIL import Image
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 16
IMAGE_SIZE = 256  
EMBEDDING_DIM = 256  
NUM_CODEBOOK_VECTORS = 1024  
BETA = 0.25  


TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_DIM = 256
CONTEXT_LENGTH = 64  

LEARNING_RATE = 3e-4
NUM_EPOCHS = 40


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()
            
    def forward(self, x):
        residual = x
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        return F.relu(x + self.projection(residual))

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        
        modules = []
        
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1))
        modules.append(nn.BatchNorm2d(hidden_dims[0]))
        modules.append(nn.ReLU())
        
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1))
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(nn.ReLU())
            modules.append(ResidualBlock(h_dim, h_dim))
            in_channels = h_dim
            
        modules.append(nn.Conv2d(in_channels, EMBEDDING_DIM, kernel_size=1))
        
        self.encoder = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.encoder(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z):
        
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        min_encoding_indices = torch.argmin(d, dim=1)
        min_encodings = torch.zeros_like(d).scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z.detach() - z_q) ** 2)
        
        loss = commitment_loss + codebook_loss
        
       
        z_q = z + (z_q - z).detach()
        
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        modules = []
        in_channels = EMBEDDING_DIM
        
        
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(hidden_dims[0]))
        modules.append(nn.ReLU())
        
        
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(nn.ConvTranspose2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1))
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(nn.ReLU())
            modules.append(ResidualBlock(h_dim, h_dim))
            in_channels = h_dim
            
        
        modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        modules.append(nn.Tanh())  
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.vector_quantizer = VectorQuantizer(
            NUM_CODEBOOK_VECTORS, EMBEDDING_DIM, BETA
        )
        self.decoder = Decoder()
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, indices
    
    def encode(self, x):
        z = self.encoder(x)
        z_q, _, indices = self.vector_quantizer(z)
        return indices
    
    def decode(self, indices):

        batch_size, height, width = indices.shape
        one_hot = torch.zeros(batch_size, NUM_CODEBOOK_VECTORS, height, width).to(indices.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        

        z_q = torch.matmul(one_hot.view(-1, NUM_CODEBOOK_VECTORS), self.vector_quantizer.embedding.weight)
        z_q = z_q.view(batch_size, height, width, EMBEDDING_DIM).permute(0, 3, 1, 2)
        
     
        x_recon = self.decoder(z_q)
        return x_recon


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, token_file, split_file=None, transform=None, max_caption_length=64):
        self.img_dir = img_dir
        self.transform = transform
        self.max_caption_length = max_caption_length
        self.captions = {}
        self.image_files = []
        
        
        split_images = None
        if split_file:
            split_images = set()
            with open(split_file, 'r') as f:
                for line in f:
                    image_id = line.strip()
                    split_images.add(image_id)
        
       
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.next_token_id = 2
        
        
        print("Loading captions and building vocabulary...")
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) == 2:
                    img_caption_id, caption = parts
                    
                    img_file = img_caption_id.split('#')[0]
                    
                    
                    if split_images is not None and img_file not in split_images:
                        continue
                    
                    if img_file not in self.captions:
                        self.captions[img_file] = []
                        self.image_files.append(img_file)
                    
                    self.captions[img_file].append(caption)
                    
                   
                    for word in caption.lower().split():
                        if word not in self.vocab:
                            self.vocab[word] = self.next_token_id
                            self.next_token_id += 1
        
        print(f"Loaded {len(self.image_files)} images with captions")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
       
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Warning: Could not load image {img_path}")
            image = Image.new('RGB', (256, 256), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
       
        caption = random.choice(self.captions[img_file])
        caption_text = caption.lower()
        
        
        caption_tokens = [self.vocab.get(word, self.vocab["<unk>"]) for word in caption_text.split()]
        
       
        if len(caption_tokens) > self.max_caption_length:
            caption_tokens = caption_tokens[:self.max_caption_length]
        else:
            caption_tokens += [self.vocab["<pad>"]] * (self.max_caption_length - len(caption_tokens))
        
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)
        
        return image, caption_tensor, caption_text
class TextToImageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=TRANSFORMER_DIM, nhead=TRANSFORMER_HEADS,
                 num_layers=TRANSFORMER_LAYERS, dim_feedforward=1024, 
                 codebook_size=NUM_CODEBOOK_VECTORS):
        super().__init__()
        
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.image_token_embedding = nn.Embedding(NUM_CODEBOOK_VECTORS, TRANSFORMER_DIM)

        
        self.bos_token = vocab_size  
        self.eos_token = vocab_size + 1  
        self.img_token = vocab_size + 2  
        
       
        self.special_token_embedding = nn.Embedding(3, d_model)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        
        self.output_projection = nn.Linear(d_model, codebook_size)
        
      
        self.img_size = IMAGE_SIZE // 4
        self.total_img_tokens = self.img_size * self.img_size
        
    def forward(self, text_indices, image_indices=None, generate=False, max_len=None):
        batch_size = text_indices.size(0)
        
        
        text_embeddings = self.token_embedding(text_indices)
        text_embeddings = self.positional_encoding(text_embeddings)
        
        
        bos_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=text_indices.device) + self.bos_token
        bos_embeddings = self.special_token_embedding(torch.zeros_like(bos_idx))
        
       
        img_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=text_indices.device) + self.img_token
        img_embeddings = self.special_token_embedding(torch.ones_like(img_idx) * 2)

        
        if generate:
            
            sequence = torch.cat([bos_embeddings, text_embeddings, img_embeddings], dim=1)
            mask = None  
            
            
            return self._generate(sequence, max_len or self.total_img_tokens)
        else:
            
            flat_img_indices = image_indices.view(batch_size, -1)
            
            
            seq_len = 2 + text_indices.size(1) + flat_img_indices.size(1) 
            mask = self._create_causal_mask(seq_len).to(text_indices.device)
            
            
            img_embeddings_projected = self.image_token_embedding(flat_img_indices)
            sequence = torch.cat([
                bos_embeddings, 
                text_embeddings, 
                img_embeddings, 
                img_embeddings_projected
            ], dim=1)
            
            transformer_output = self.transformer_encoder(sequence, mask=mask)
            
           
            relevant_output = transformer_output[:, 2+text_indices.size(1):-1]  
            logits = self.output_projection(relevant_output)
            
            
            targets = flat_img_indices
            
            return logits, targets
            
    def _create_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def _generate(self, initial_sequence, max_tokens):
        device = initial_sequence.device
        batch_size = initial_sequence.size(0)
        curr_len = initial_sequence.size(1)
        
        output_indices = torch.zeros(batch_size, max_tokens, dtype=torch.long, device=device)
        
        for i in range(max_tokens):
            
            mask = self._create_causal_mask(curr_len).to(device)
            
            
            transformer_output = self.transformer_encoder(initial_sequence, mask=mask)
            
            
            next_token_logits = self.output_projection(transformer_output[:, -1])
            temperature = 1.0  
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            

            
            
            output_indices[:, i] = next_token.squeeze(1)
            
            
            next_token_embedding = self.image_token_embedding(next_token)

            initial_sequence = torch.cat([initial_sequence, next_token_embedding], dim=1)
            curr_len += 1
            
        
        return output_indices.view(batch_size, self.img_size, self.img_size)




def train_vqvae(vqvae, dataloader, optimizer, num_epochs):
    vqvae.train()
    
    for epoch in range(num_epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        
        for batch_idx, (imgs, _, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            recons, vq_loss, _ = vqvae(imgs)
            
            
            recon_loss = F.mse_loss(recons, imgs)
            
            
            total_loss = recon_loss + vq_loss
            
            total_loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Recon Loss = {recon_loss.item():.4f}, VQ Loss = {vq_loss.item():.4f}")
        
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Avg Recon Loss = {avg_recon_loss:.4f}, Avg VQ Loss = {avg_vq_loss:.4f}")
        
        
        if (epoch + 1) % 5 == 0:
            vqvae.eval()
            with torch.no_grad():
                sample_imgs = next(iter(dataloader))[0][:8].to(device)
                recons, _, _ = vqvae(sample_imgs)
                
                
                sample_imgs = sample_imgs.cpu().numpy()
                recons = recons.cpu().numpy()
                
                
                sample_imgs = (sample_imgs + 1) / 2
                recons = (recons + 1) / 2
                
               
                plt.figure(figsize=(12, 6))
                for i in range(8):
                  
                    plt.subplot(2, 8, i+1)
                    plt.imshow(np.transpose(sample_imgs[i], (1, 2, 0)))
                    plt.axis('off')
                    
                   
                    plt.subplot(2, 8, 8+i+1)
                    plt.imshow(np.transpose(recons[i], (1, 2, 0)))
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"vqvae_reconstruction_epoch_{epoch+1}.png")
                plt.close()
            
            vqvae.train()
    
    return vqvae

def train_transformer(transformer, vqvae, dataloader, optimizer, num_epochs, checkpoint_path= None, start_epoch = 0):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔁 Resuming from epoch {start_epoch}")

    transformer.train()
    vqvae.eval()


    scaler = torch.amp.GradScaler()  

    try:
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0

            for batch_idx, (imgs, captions, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                imgs = imgs.to(device)
                captions = captions.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    image_indices = vqvae.encode(imgs)

                batch_size, h, w = image_indices.shape
                expected_tokens = transformer.img_size * transformer.img_size

                if h * w != expected_tokens:
                    image_indices = F.interpolate(
                        image_indices.unsqueeze(1).float(),
                        size=(transformer.img_size, transformer.img_size),
                        mode='nearest'
                    ).squeeze(1).long()

                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits, targets = transformer(captions, image_indices)

                    min_len = min(logits.reshape(-1, NUM_CODEBOOK_VECTORS).shape[0], targets.reshape(-1).shape[0])
                    logits_flat = logits.reshape(-1, NUM_CODEBOOK_VECTORS)[:min_len]
                    targets_flat = targets.reshape(-1)[:min_len]

                    loss = F.cross_entropy(logits_flat, targets_flat)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}: Loss = {loss.item():.4f}")

            print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(dataloader):.4f}")

            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
            print(f" Saved checkpoint: checkpoint_epoch_{epoch}.pt")

    except Exception as e:
        print(f" Training crashed at epoch {epoch} due to: {str(e)}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"crash_checkpoint_epoch_{epoch}.pt")
        print(f" Saved crash checkpoint at epoch {epoch}.")

        return transformer
    return transformer



def train_text_to_image_model(img_dir, token_file, train_split_file, val_split_file=None, 
                              batch_size=BATCH_SIZE, vqvae_epochs= 25, transformer_epochs=3):
   
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
    train_dataset = Flickr8kDataset(img_dir, token_file, train_split_file, transform=transform)
    
    
    val_dataset = None
    if val_split_file:
        val_dataset = Flickr8kDataset(img_dir, token_file, val_split_file, transform=transform)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    vqvae = VQVAE().to(device)
    vqvae_optimizer = optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
    
    print("Training VQ-VAE...")
    vqvae = train_vqvae(vqvae, train_dataloader, vqvae_optimizer, vqvae_epochs)
    vqvae.load_state_dict(torch.load("vqvae_model.pth"))
    
    
    torch.save(vqvae.state_dict(), "vqvae_model.pth")
    
    
    vocab_size = len(train_dataset.vocab)
    transformer = TextToImageTransformer(vocab_size).to(device)
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    
    print("Training Transformer...")
    transformer = train_transformer(transformer, vqvae, train_dataloader, transformer_optimizer, transformer_epochs)
    
    
    torch.save(transformer.state_dict(), "transformer_model.pth")
    
    return vqvae, transformer, train_dataset.vocab


def generate_image_from_text(text, vqvae, transformer, vocab):
    vqvae.eval()
    transformer.eval()
    
    
    tokens = []
    for word in text.split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(vocab["<unk>"])
    
    
    if len(tokens) > CONTEXT_LENGTH:
        tokens = tokens[:CONTEXT_LENGTH]
    else:
        tokens += [vocab["<pad>"]] * (CONTEXT_LENGTH - len(tokens))
    
    
    text_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
  
    with torch.no_grad(), torch.amp.autocast(device_type= "cuda"):
        generated_indices = transformer(text_tensor, generate=True)
        generated_indices = generated_indices.view(1, 64, 64)
        
        
        generated_img = vqvae.decode(generated_indices)
        
        
        generated_img = generated_img.cpu().numpy()
        
        
        generated_img = (generated_img[0] + 1) / 2
        generated_img = np.transpose(generated_img, (1, 2, 0))
        generated_img = generated_img.astype(np.float32)
    
    return generated_img


if __name__ == "__main__":
    
    img_dir = "./New folder/Flicker8k_Dataset"  
    token_file = "./New folder/Flickr8k.token.txt"
    train_split_file = "./New folder/Flickr_8k.trainImages.txt"
    val_split_file = "./New folder/Flickr_8k.devImages.txt"

    vqvae, transformer, vocab = train_text_to_image_model(
    img_dir, 
    token_file,
    train_split_file,
    val_split_file
    )
    

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
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(prompt)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("generated_samples.png")           
    plt.show()