"""
Show CIFAR-10 dataset samples.
Run as:
python -m myscripts.cifar.show_data
"""

from torchvision import datasets, transforms
from PIL import Image

def print_image_ascii(img_tensor, width=64):
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

def show_data():
    transform = transforms.ToTensor()
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    class_names = train_dataset.classes
    
    for i in range(5):
        image, label = train_dataset[i]
        print(f"\nItem {i}: Image size: {image.shape}, Label: {label} ({class_names[label]})")
        print_image_ascii(image)

if __name__ == "__main__":
    show_data()
