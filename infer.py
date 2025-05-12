import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载验证集
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# 加载模型
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(val_dataset.classes))
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# 推理
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        print(f"Predicted: {val_dataset.classes[preds.item()]}, Actual: {val_dataset.classes[labels.item()]}")
