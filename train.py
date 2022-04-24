from torchvision import transforms
from torchvision import datasets
import torch
from MLP.mlp import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

input_size = 28 * 28
hidden_size = 512
num_classes = 10
num_epoch = 200
batch_size = 64

train_data = datasets.MNIST(root='./dataset/mnist', train=True, transform = transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='.dataset/mnist', train=False, transform = transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model = mlp5(input_size, hidden_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x_list = []
y_list = []

for epoch in range(num_epoch):
  for batch_idx, (x, y) in enumerate(train_loader):
    x = x.reshape(-1, 28*28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y = y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if batch_idx % 200 == 0:
      print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch + 1, num_epoch, batch_idx + 1, loss.data.item()))
    if batch_idx == 0:
        x_list.append(epoch)
        y_list.append(loss.data.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure()
plt.plot(x_list,y_list)
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('mlp5-SGD-CELoss')
plt.savefig('mlp5SGDCELoss.jpg')
plt.show()
state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
torch.save(state, 'ckpt/mlp5SGDCELoss.pth')
