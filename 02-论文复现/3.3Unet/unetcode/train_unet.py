import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子以保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置设备
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 定义超参数
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 2  # 背景 + 肺部
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# 定义VOC数据集类
class VOCSegmentationDataset(Dataset):
    def __init__(self, voc_root, image_set='train', transform=None):
        self.voc_root = voc_root
        self.image_set = image_set
        self.transform = transform
        
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClassPNG')
        
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        
        with open(split_file, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, f'{x}.jpg') for x in file_names]
        self.masks = [os.path.join(mask_dir, f'{x}.png') for x in file_names]
        
        assert len(self.images) == len(self.masks)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # 读取图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图像
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
            # 调整掩码大小并转换为张量
            mask = mask.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

# 定义UNet模型的双卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        
        # 编码器（下采样路径）
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器（上采样路径）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # 最终分类层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码器路径
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

# 定义损失函数
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        # 将输入从 [N, C, H, W] 转换为 [N, C, H*W]
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# 定义评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算IOU
            preds = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(preds == 1, masks == 1).sum().item()
            union = torch.logical_or(preds == 1, masks == 1).sum().item()
            iou = intersection / union if union > 0 else 0
            
            running_loss += loss.item() * images.size(0)
            running_iou += iou * images.size(0)
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    
    return epoch_loss, epoch_iou

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        # 训练
        loop = tqdm(train_loader, leave=True)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            preds = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(preds == 1, masks == 1).sum().item()
            union = torch.logical_or(preds == 1, masks == 1).sum().item()
            iou = intersection / union if union > 0 else 0
            
            running_loss += loss.item() * images.size(0)
            running_iou += iou * images.size(0)
            
            # 更新进度条
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), iou=iou)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_iou = running_iou / len(train_loader.dataset)
        
        # 验证
        val_loss, val_iou = evaluate(model, val_loader, criterion)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        train_ious.append(epoch_train_iou)
        val_ious.append(val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, Train IoU: {epoch_train_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"保存最佳模型，验证IoU: {val_iou:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Training and Validation IoU')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, train_losses, val_losses, train_ious, val_ious

# 可视化预测结果
def visualize_predictions(model, dataloader, num_samples=3):
    model.eval()
    
    for i, (images, masks) in enumerate(dataloader):
        if i >= num_samples:
            break
            
        with torch.no_grad():
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
        
        # 转换为numpy数组
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        for j in range(min(images.shape[0], 3)):
            plt.subplot(3, 3, j*3+1)
            plt.imshow(images[j])
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(3, 3, j*3+2)
            plt.imshow(masks[j], cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(3, 3, j*3+3)
            plt.imshow(preds[j], cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'prediction_samples_{i}.png')
        plt.show()

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = VOCSegmentationDataset(voc_root='VOCDatasets', image_set='train', transform=transform)
    val_dataset = VOCSegmentationDataset(voc_root='VOCDatasets', image_set='val', transform=transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建模型、损失函数和优化器
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    model, train_losses, val_losses, train_ious, val_ious = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS
    )
    
    # 加载最佳模型并可视化预测结果
    model.load_state_dict(torch.load('best_unet_model.pth'))
    visualize_predictions(model, val_loader)

if __name__ == '__main__':
    main() 