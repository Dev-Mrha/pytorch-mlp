from torchvision import transforms
from torchvision import datasets
import torch
from MLP.mlp import mlp3
from torch.utils.data import DataLoader

input_size = 28 * 28
hidden_size = 512
num_classes = 10
num_epoch = 50
batch_size = 64

train_data = datasets.MNIST(root='./dataset/mnist', train=True, transform = transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='.dataset/mnist', train=False, transform = transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model = mlp3(input_size, hidden_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epoch):
  for batch_idx, (x, y) in enumerate(train_loader):
    x = x.reshape(-1, 28*28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y = y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if batch_idx % 200 == 1:
      print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch + 1, num_epoch, batch_idx + 1, loss.data.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
torch.save(state, 'ckpt/mlp3SGDCELoss.pth')
