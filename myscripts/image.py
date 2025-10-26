"""
Image categorization training using GPT model.
Run as:
python -m myscripts.image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nanochat.gpt import GPT, GPTConfig

def create_image_classifier(num_classes=10, image_size=28):
    config = GPTConfig(
        sequence_len=image_size * image_size,
        vocab_size=256,
        n_layer=6,
        n_head=4,
        n_kv_head=4,
        n_embd=256
    )
    
    model = GPT(config)
    model.init_weights()
    
    model.classifier_head = nn.Linear(config.n_embd, num_classes)
    nn.init.zeros_(model.classifier_head.weight)
    nn.init.zeros_images(model.classifier_head.bias)
    
    return model

def train_image_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long().squeeze())
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = create_image_classifier(num_classes=10, image_size=28)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                x = model.transformer.wte(images)
                x = F.rms_norm(x, (x.size(-1),))
                
                cos_sin = model.cos[:, :images.size(1)], model.sin[:, :images.size(1)]
                for block in model.transformer.h:
                    x = block(x, cos_sin, kv_cache=None)
                x = F.rms_norm(x, (x.size(-1),))
                
                x = x.mean(dim=1)
                logits = model.classifier_head(x)
                
                loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {100.*correct/total:.2f}%")
        
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                    x = model.transformer.wte(images)
                    x = F.rms_norm(x, (x.size(-1),))
                    
                    cos_sin = model.cos[:, :images.size(1)], model.sin[:, :images.size(1)]
                    for block in model.transformer.h:
                        x = block(x, cos_sin, kv_cache=None)
                    x = F.rms_norm(x, (x.size(-1),))
                    
                    x = x.mean(dim=1)
                    logits = model.classifier_head(x)
                
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        print(f"Test Acc: {100.*test_correct/test_total:.2f}%\n")

if __name__ == "__main__":
    train_image_classifier()
