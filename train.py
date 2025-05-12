import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt   # 新增：导入画图库
from utils import get_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_classes = 5  # 类别数
num_epochs = 5
batch_size = 8
learning_rate = 1e-3
data_dir = './data'

# 1. 加载数据
train_loader, val_loader = get_data_loaders(data_dir, batch_size)

# 2. 加载预训练模型
model = models.mobilenet_v3_small(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # 冻结特征提取层
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes) #替换分类头
model = model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练循环 + 记录loss
train_losses = []  # 新增：记录每个epoch的loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)  # 记录每一轮的loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 5. 绘制训练loss曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('loss_curve.png')  # 保存成图片
plt.show()

# 6. 保存训练好的模型
torch.save(model.state_dict(), 'model.pth')
print("训练完成，模型保存为 model.pth")
