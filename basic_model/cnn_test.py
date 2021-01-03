import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 2D卷积层。输入通道1，输出通道16（16个卷积核），卷积核大小为5*5，步长1，补全2
                                                                   # 即卷积核有1个通道（简要地，5*5*1）
                                                                   # 输入: (10, 1, 28, 28)  输出: (10, 16, 28, 28)
            nn.BatchNorm2d(16),  # 批规范化 shape不变
            nn.ReLU(),  # 激活
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层  (10, 16, 28, 28)->(10, 16, 14, 14)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # (10, 16, 14, 14)->(10, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # (10, 32, 14, 14)->(10, 32, 7, 7)
        
        self.fc = nn.Linear(7*7*32, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()   #使用交叉熵作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)	# 优化器，用于在反向传播中进行参数更新

input_data = torch.randn(10,1,28,28).to(device)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))