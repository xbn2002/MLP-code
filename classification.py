import torch
import torchvision
from torchvision import transforms
from torch import nn
from numpy import *
from torch.autograd import Variable

# 多层神经网络-多分类问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 用gpu跑，速度快
print(device)

# 超参数设置
EPOCH = 2  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率（更新步长），不能过大（导致梯度爆炸、loss激增），不能过小（收敛慢）

# 定义数据预处理方式
transform = transforms.ToTensor()  # 把PIL图像或np数组转为tensor

# 定义训练数据集
trainset = torchvision.datasets.MNIST(
    root = '../../data',
    train = True,  # 是否训练
    download = True,
    transform = transform  #预处理方式
)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size = BATCH_SIZE,  # batch大小
    shuffle = False,  # 是否打乱次序
)

# 定义测试数据集
testset = torchvision.datasets.MNIST(
    root = '../../data',
    train = False,
    download = True,
    transform = transform
)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size = BATCH_SIZE,
    shuffle = False,
)


class module_net(nn.Module):  # 网络定义
    def __init__(self):  # 可以在init里预定义一些层
        super().__init__()
        # 以下写法是无序的
        self.layer1 = nn.Linear(784, 400)
        self.layer2 = nn.ReLU()
        # nn.Sigmoid()
        # nn.Tanh()
        self.layer3 = nn.Linear(400, 200)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(200, 100)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(100, 10)
        # 以下写法是有序的
        self.layer10 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):  # 定义正向传播方向，即layer之间的排列顺序
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # or:
        # x = self.layer10(x)
        return x


model = module_net()  # 实例化
optimizer = torch.optim.SGD(model.parameters(), lr=LR)  # 定义优化方法：随机梯度下降法
criterion = nn.CrossEntropyLoss()  # 定义损失计算方法：交叉熵-多分类的损失函数
for epoch in range(EPOCH):
    for i, data in enumerate(trainloader):
        # 数据提取
        inputs, labels = data

        a1, a2, a3, a4 = shape(inputs) #np的shape方法，返回一个元组：n维数组，从前到后依次为从高到低维数分别有几个元素

        inputs = inputs.view(a1, a3 * a4) # a1是batch_size，a3,a4为长和宽；a2是图片通道数（参数个数）
        inputs = Variable(inputs)  #从torch里autograd引入，为了自动计算梯度
        labels = Variable(labels)

        # 前向传播，输出prediction并计算loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 根据预测值和实际值，计算损失

        # 反向传播，根据loss对参数进行修正
        loss.backward()  # 计算残差反向传播
        optimizer.step()  # 执行参数更新
        optimizer.zero_grad()  # 梯度归零，为了阻止自动叠加

        if(i + 1) % 32 == 0:
            print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, EPOCH, loss.data.item())) # 每跑32次就输出一下当前的loss值

    # 每跑完一次epoch测试一下准确率
    with torch.no_grad():  # 测试的时候无需计算梯度
        correct = 0
        total = 0

        for data in testloader:  # 遍历数据集
            images, labels = data

            a1, a2, a3, a4 = shape(images)
            inputs = images.view(a1, 784)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)  # 取得分最高的那个类

            # 这里把正确率作为衡量指标
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))

