from torchvision import transforms
from torchvision import datasets
import torch
from MLP.mlp import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

input_size = 28 * 28
num_classes = 10
num_epoch = 150
batch_size = 64

train_data = datasets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='.dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

x_list = []
y_list = []
z_list = []


def train(epoch, model, criterion, optimizer):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.reshape(-1, 28 * 28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if batch_idx % 200 == 0:
            print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch + 1, num_epoch, batch_idx + 1, loss.data.item()))
        if batch_idx == 0:
            y_list.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            image, labels = data
            image = image.reshape(-1, 28 * 28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    z_list.append(100.0 * correct / total)
    print('Accuracy on test set: %.6lf %%' % (100.0 * correct / total))


cnt = 0


def trainn(model, lr, name, pth):
    # global cnt
    # global plt
    txtName = name + ".txt"
    f = open(txtName, "a+")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global x_list
    global y_list
    global z_list
    x_list = []
    y_list = []
    z_list = []
    for i in range(num_epoch):
        x_list.append(i)
        train(i, model, criterion, optimizer)
        test(model)
    f.write(str(y_list))
    f.write("\n")
    f.write(str(z_list))
    # cnt = cnt + 1
    # pic = plt.figure(cnt)
    # plt, ax = plt.subplots()
    # ax.plot(x_list,y_list,"r")
    # ax.set_xlabel('epoch')
    # ax.set_ylabel('loss')
    # ax.set_title(name)
    # ax2 = ax.twinx()
    # ax2.plot(x_list,z_list,"g")
    # ax2.set_ylabel('accuracy')

    # plt.savefig(name+'.jpg')
    # plt.show()
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epoch}
    torch.save(state, pth)


model = mlp3(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.01, name='mlp3-Adam-0.01', pth='ckpt/mlp3Adam001.pth')

model = mlp3(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.001, name='mlp3-Adam-0.001', pth='ckpt/mlp3Adam0001.pth')

model = mlp5(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.01, name='mlp5-Adam-0.01', pth='ckpt/mlp5Adam001.pth')

model = mlp5(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.001, name='mlp5-Adam-0.001', pth='ckpt/mlp5Adam0001.pth')

model = mlp7(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.01, name='mlp7-Adam-0.01', pth='ckpt/mlp7Adam001.pth')

model = mlp7(input_size, num_classes)
model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
trainn(model=model, lr=0.001, name='mlp7-Adam-0.001', pth='ckpt/mlp7Adam0001.pth')