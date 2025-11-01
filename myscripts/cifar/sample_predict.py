"""
Run predictions on sample CIFAR-10 images.
Run as:
python -m myscripts.cifar.sample_predict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = x.float() / 255.0
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x

def create_patch_classifier(num_classes=10, image_size=32, patch_size=4):
    num_patches = (image_size // patch_size) ** 2
    
    config = GPTConfig(
        sequence_len=num_patches,
        vocab_size=1,
        n_layer=6,
        n_head=4,
        n_kv_head=4,
        n_embd=256
    )
    
    model = GPT(config)
    model.init_weights()
    
    model.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=config.n_embd)
    model.classifier_head = nn.Linear(config.n_embd, num_classes)
    nn.init.zeros_(model.classifier_head.weight)
    nn.init.zeros_(model.classifier_head.bias)
    
    return model

def print_image_ascii(img_tensor, width=32):
    img = transforms.ToPILImage()(img_tensor)
    w, h = img.size
    aspect_ratio = h / w
    new_height = int(aspect_ratio * width * 0.5)
    img = img.resize((width, new_height * 2))
    img = img.convert('RGB')

    for y in range(0, new_height * 2, 2):
        for x in range(width):
            r1, g1, b1 = img.getpixel((x, y))
            r2, g2, b2 = img.getpixel((x, y + 1))
            print(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m", end="")
        print()

def sample_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_dir = ".checkpoints/cifar10_by_patch"
    
    from nanochat.checkpoint_manager import find_last_step
    step = find_last_step(checkpoint_dir)
    
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    
    model = create_patch_classifier(num_classes=10, image_size=32, patch_size=4)
    model.load_state_dict(model_data)
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long())
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    class_names = test_dataset.classes
    
    with torch.no_grad():
        for i in range(5):
            image, label = test_dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                x = model.patch_embed(image_batch)
                x = F.rms_norm(x, (x.size(-1),))
                
                cos_sin = model.cos[:, :x.size(1)], model.sin[:, :x.size(1)]
                for block in model.transformer.h:
                    x = block(x, cos_sin, kv_cache=None)
                x = F.rms_norm(x, (x.size(-1),))
                
                x = x.mean(dim=1)
                logits = model.classifier_head(x)
            
            predicted = logits.argmax(1).item()
            
            display_image = image.float() / 255.0
            print(f"\nSample {i}:")
            print(f"True Label: {label} ({class_names[label]})")
            print(f"Predicted: {predicted} ({class_names[predicted]})")
            print_image_ascii(display_image)

if __name__ == "__main__":
    sample_predict()
