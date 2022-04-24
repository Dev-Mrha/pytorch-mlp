from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from MLP.mlp import *
import torch
import numpy as np


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


input_size = 28 * 28
hidden_size = 512
num_classes = 10
num_epoch = 50
batch_size = 64

test_data = datasets.MNIST(root='.dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, shuffle=False)

model = mlp5(input_size, hidden_size, num_classes).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
checkpoint = torch.load('ckpt/mlp5SGDCELoss.pth')
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch'] + 1

def test():
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
            correct += (predicted==labels).sum().item()
    print('Accuracy on test set: %.6lf %%' % (100.0*correct/total))
'''
for batch_idx, (x, y) in enumerate(test_loader):
    x = x.reshape(-1, 28 * 28).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y = y.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_pred = model(x)
    print('acc: ', accuracy_score(y, y_pred))

'''
test()