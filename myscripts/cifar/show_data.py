"""
Show CIFAR-10 dataset samples.
Run as:
python -m myscripts.cifar.show_data
"""

from torchvision import datasets, transforms

def show_data():
    transform = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    class_names = train_dataset.classes
    
    for i in range(5):
        image, label = train_dataset[i]
        print(f"Item {i}: Image size: {image.shape}, Label: {label} ({class_names[label]})")

if __name__ == "__main__":
    show_data()
