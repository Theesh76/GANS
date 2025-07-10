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
image_size = 64
batch_size = 8
noise_dim = 100
epochs = 100
lr = 0.0002
best_loss_G = float('inf')
patience = 50
epochs_no_improve = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GANS_result_save_path = 'C:/Users/sathi/Research/GANS_Result/'
data_path = np.array(glob.glob("C:/Users/sathi/OneDrive/Desktop/Aadhan_8_picz/*.png") + glob.glob("C:/Users/sathi/OneDrive/Desktop/Aadhan_8_picz/*.jpg"))  # <-- SET YOUR CUSTOM DATA PATH HERE

# ================================
# TRANSFORM + DATALOADER
# ================================
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
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
    G.train()
    D.train()
    
    loss_G_epoch = 0.0
    loss_D_epoch = 0.0
    num_batches = 0

    for real_imgs in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_imgs = G(noise)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        out_real = D(real_imgs)
        out_fake = D(fake_imgs.detach())

        loss_real = criterion(out_real, real_labels)
        loss_fake = criterion(out_fake, fake_labels)
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator ===
        fake_imgs = G(noise)
        out_fake = D(fake_imgs)
        loss_G = criterion(out_fake, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Accumulate losses
        loss_G_epoch += loss_G.item()
        loss_D_epoch += loss_D.item()
        num_batches += 1

    # Calculate average loss for the epoch
    avg_loss_G = loss_G_epoch / num_batches
    avg_loss_D = loss_D_epoch / num_batches

    # === Save model if Generator improved ===
    if avg_loss_G < best_loss_G:
        best_loss_G = avg_loss_G
        epochs_no_improve = 0

        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'loss_G': best_loss_G
        }, os.path.join(GANS_result_save_path, "best_model" + str(epoch) + ".pth"))

        print(f"✅ Saved improved model at epoch {epoch+1}, avg_loss_G: {best_loss_G:.4f}")

        # Save sample image
        with torch.no_grad():
            test_z = torch.randn(16, noise_dim, device=device)
            test_imgs = G(test_z)
            grid = utils.make_grid(test_imgs, nrow=4, normalize=True)
            img_path = os.path.join(GANS_result_save_path, f"epoch_{epoch+1}.png")
            utils.save_image(grid, img_path)
            print(f"🖼️ Saved sample image to {img_path}")

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered. Training stopped.")
            break

    print(f"📘 Epoch [{epoch+1}/{epochs}]  Avg Loss_D: {avg_loss_D:.4f}, Avg Loss_G: {avg_loss_G:.4f}")