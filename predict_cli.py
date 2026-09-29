import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# Definisikan arsitektur yang sama dengan di notebook
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"File model '{model_path}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan sel 'torch.save(model.state_dict(), ...)' di pytorch.ipynb")
        return

    # Load model
    model = CNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Gagal memuat model dari {model_path}. Error: {e}")
        return

    # Transformasi gambar
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Gagal memuat gambar dari {image_path}. Error: {e}")
        return

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    print(f"--- Hasil Prediksi ---")
    print(f"Gambar: {image_path}")
    print(f"Prediksi Kelas: {classes[predicted[0].item()].upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediksi Gambar menggunakan CNN CIFAR-10')
    parser.add_argument('--image', type=str, required=True, help='Path ke gambar uji')
    parser.add_argument('--model', type=str, default='model_cifar10.pth', help='Path ke model berformat .pth')
    
    args = parser.parse_args()
    predict(args.image, args.model)
