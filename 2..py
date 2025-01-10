import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义神经网络模型
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 创建一些示例数据
X = torch.randn(100, 10)  # 100个样本，每个样本10个特征
y = torch.randint(0, 2, (100,))  # 二分类问题的标签

# 初始化模型、损失函数和优化器
model = SimpleNeuralNetwork(input_size=10, hidden_size=20, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X.float())
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_input = torch.randn(10, 10)  # 10个测试样本
    predictions = model(test_input.float())
    _, predicted = torch.max(predictions.data, 1)
    print("\n测试预测结果:", predicted)
