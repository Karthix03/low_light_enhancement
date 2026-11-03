"""
Train Zero-DCE (Zero-Reference Deep Curve Estimation) Neural Network
Trains the 7-layer convolutional illumination curve network on low-light frames
using 4 unsupervised non-reference loss functions:
  1. L_spa: Spatial Consistency Loss
  2. L_exp: Exposure Control Loss
  3. L_col: Color Constancy Loss
  4. L_tv:  Illumination Smoothness Loss
Saves trained PyTorch weights to models/zero_dce.pth.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "zero_dce.pth")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "datasets", "sample_night_traffic", "train", "images")

# ============================================================================
# Zero-DCE Network Architecture
# ============================================================================
class ZeroDCENetwork(nn.Module):
    def __init__(self, number_f=32, n_curves=8):
        super(ZeroDCENetwork, self).__init__()
        self.n_curves = n_curves
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        
        self.conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(number_f * 2, 3 * n_curves, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
        
        r_list = torch.split(r, 3, dim=1)
        x_enhanced = x
        for i in range(self.n_curves):
            x_enhanced = x_enhanced + r_list[i] * x_enhanced * (1.0 - x_enhanced)
            
        return torch.clamp(x_enhanced, 0.0, 1.0), r

# ============================================================================
# Zero-Reference Loss Functions
# ============================================================================
class L_color(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        return torch.mean(torch.sqrt(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2)))

class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean = self.pool(torch.mean(x, 1, keepdim=True))
        return torch.mean(torch.pow(mean - self.mean_val, 2))

class L_tv(nn.Module):
    def forward(self, r):
        batch_size = r.size()[0]
        h_x = r.size()[2]
        w_x = r.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((r[:, :, 1:, :] - r[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        org_mean = torch.mean(org, 1, keepdim=True)
        enh_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)

        d_org_l = F.conv2d(org_pool, self.weight_left.to(org.device), padding=1)
        d_org_r = F.conv2d(org_pool, self.weight_right.to(org.device), padding=1)
        d_org_u = F.conv2d(org_pool, self.weight_up.to(org.device), padding=1)
        d_org_d = F.conv2d(org_pool, self.weight_down.to(org.device), padding=1)

        d_enh_l = F.conv2d(enh_pool, self.weight_left.to(org.device), padding=1)
        d_enh_r = F.conv2d(enh_pool, self.weight_right.to(org.device), padding=1)
        d_enh_u = F.conv2d(enh_pool, self.weight_up.to(org.device), padding=1)
        d_enh_d = F.conv2d(enh_pool, self.weight_down.to(org.device), padding=1)

        d_l = torch.pow(d_org_l - d_enh_l, 2)
        d_r = torch.pow(d_org_r - d_enh_r, 2)
        d_u = torch.pow(d_org_u - d_enh_u, 2)
        d_d = torch.pow(d_org_d - d_enh_d, 2)
        return torch.mean(d_l + d_r + d_u + d_d)

# ============================================================================
# Dataset
# ============================================================================
class LowLightDataset(Dataset):
    def __init__(self, img_dir, size=(256, 256)):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)

# ============================================================================
# Training Execution
# ============================================================================
def main():
    print("\n" + "=" * 65)
    print("      NIGHTVISION AI - ZERO-DCE ENHANCER TRAINING")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset = LowLightDataset(TRAIN_IMG_DIR)
    if len(dataset) == 0:
        print(f"[ERROR] No training images found in {TRAIN_IMG_DIR}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset Size: {len(dataset)} low-light images")

    model = ZeroDCENetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_spa = L_spa()
    loss_exp = L_exp(16, 0.6)
    loss_col = L_color()
    loss_tv = L_tv()

    epochs = 10
    print(f"Training Zero-DCE for {epochs} epochs...")
    print("-" * 65)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            enhanced, r = model(batch)
            
            l_spa = loss_spa(batch, enhanced)
            l_exp = 10.0 * loss_exp(enhanced)
            l_col = 5.0 * loss_col(enhanced)
            l_tv = 200.0 * loss_tv(r)
            
            total_loss = l_spa + l_exp + l_col + l_tv
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        avg_loss = running_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} (Spa: {l_spa.item():.4f}, Exp: {l_exp.item():.4f}, Col: {l_col.item():.4f}, TV: {l_tv.item():.4f})")

    # Save weights
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("-" * 65)
    print(f"[SUCCESS] Zero-DCE model successfully saved to: {OUTPUT_PATH}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
