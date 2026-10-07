import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 检测你的 RTX 5060！
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用: {device}")

# 2. 定义一个极简的神经网络（理解：就是一层“隐层”）
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)  # 28*28=784个像素输入，输出10个数字(0-9)
    def forward(self, x):
        x = x.view(-1, 28*28)  # 把图片拉直成一维
        return self.fc(x)

# 3. 加载数据（会自动下载到当前目录，约 60MB）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. 初始化模型、损失函数、优化器
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5. 开始训练（只跑 2 个 epoch，很快就能看到效果）
print("开始训练...")
for epoch in range(2):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, 平均损失: {total_loss/len(train_loader):.4f}")

print("训练完成！你的显卡成功跑起来了！")