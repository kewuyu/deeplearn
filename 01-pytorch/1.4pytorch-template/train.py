import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.Net import Net
from data import dataloader
import argparse
from tqdm import tqdm
from torch import nn
from utils.Metric import Metric
import numpy as np  # 添加numpy导入

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train AlexNet')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--root', type=str, default='./data')
args = parser.parse_args()

# 参数设置
num_classes = args.num_classes
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs

# 数据加载
train_loader, val_loader, test_loader = dataloader(root=args.root,batch_size=batch_size)

# 模型初始化
model = Net(num_classes=num_classes)

# 设备选择
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard可视化
writer = SummaryWriter()



# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 遍历训练数据
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
        images, labels = images.to(device), labels.to(device)  # 将数据迁移到GPU

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 记录训练集的预测和真实标签
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    # 使用Metric计算训练准确度
    train_accuracy = Metric(correct_train, total_train).accuracy()
    train_loss = running_loss / len(train_loader)

    # 分别记录损失和准确度
    writer.add_scalar('Loss/Train Loss', train_loss, epoch)
    writer.add_scalar('Accuracy/Train Accuracy', train_accuracy, epoch)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}')

    # 验证模型
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # 在验证时不计算梯度
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    # 使用Metric计算验证准确度
    val_accuracy = Metric(correct_val, total_val).accuracy()
    val_loss = val_running_loss / len(val_loader)

    # 分别记录验证损失和准确度
    writer.add_scalar('Loss/Validation Loss', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation Accuracy', val_accuracy, epoch)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


    # 测试模型
    model.eval()
    correct_test = 0
    total_test = 0
    test_running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    # 使用Metric计算测试准确度
    test_accuracy = Metric(correct_test, total_test).accuracy()
    test_loss = test_running_loss / len(test_loader)

    # 分别记录测试损失和准确度
    writer.add_scalar('Loss/Test Loss', test_loss, epoch)
    writer.add_scalar('Accuracy/Test Accuracy', test_accuracy, epoch)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 关闭TensorBoard writer
writer.close()
