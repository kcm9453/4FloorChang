import torch
import torch.nn
import numpy as np
import torchvision

# x = torch.tensor(1., requires_grad=True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)

# y = x*w + b # y= 2 * x + b

# y.backward()

# print(x.grad, w.grad, b.grad)

# x = torch.randn(10,3)
# y = torch.randn(10,2)

# linear = nn.linear(3,2)
# print('w: ', linear.weight)
# print('b: ', linear.bias)

# criterion = nn.MSELoss() # Mean Square Error Loss
# optimizer = torch.optim.SGD(Linear.parameters(), lr=0.01)

# for i in range(100):
   # pred = linear(x)
   # loss = criterion(pred, y)
   # print('loss : ', loss.item())
   # loss.backward()
   # print('dL/dw:', linear.weight.gred)
   # print('dL/db:', linear.bias.gred)
   # optimizer.step()
# pred = linear(x)
# loss = criterion(pred, y)
# print(loss.item())

# x = np.array([[1,2],[3,4]])
# y = torch.from_numpy(x)
# z = y.numpy()
# print("A")

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

image, label = train_dataset[0]
print(image.size())
print(label)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size=64,
    shuffle=True
)

data_iter = iter(train_loader)

image, labels = data_iter.next()

for image, labels in train_loader:
    pass