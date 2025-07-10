import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import glob as glob
import cv2
import numpy as np

class Load_Custom_Dataset(Dataset):
    def __init__(self, image_file_names, transform):
        self.image_file_names = image_file_names
        self.transform = transform
    def __len__(self):
        return len(self.image_file_names)
    def __getitem__(self, index):
        bgr_image = cv2.imread(self.image_file_names[index])
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image) 
        if self.transform:
            transformed_img = self.transform(pil_image)
        print(transformed_img.shape)
        return transformed_img

# ================================
# GENERATOR
# ================================
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ================================
# DISCRIMINATOR
# ================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ================================
# CONFIG
# ================================
image_size = 256
batch_size = 8
noise_dim = 100
epochs = 100
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = np.array(glob.glob("C:/Users/sathi/OneDrive/Desktop/Aadhan_8_picz/*.png") + glob.glob("C:/Users/sathi/OneDrive/Desktop/Aadhan_8_picz/*.jpg"))  # <-- SET YOUR CUSTOM DATA PATH HERE

# ================================
# TRANSFORM + DATALOADER
# ================================
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # to [-1, 1]
])

dataset = Load_Custom_Dataset(data_path, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
tensor_image = next(iter(dataloader))


# ================================
# INIT
# ================================
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ================================
# TRAINING LOOP
# ================================
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        noise = torch.rand