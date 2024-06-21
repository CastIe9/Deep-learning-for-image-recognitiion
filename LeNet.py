import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

path = '/Downloads'

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

    def forward(self, input):
        input = self.conv1(input)
        input = F.relu(input)
        return input


class S2(nn.Module):
    def __init__(self):
        super(S2, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size = 2, stride =2)

    def forward(self, input):
        input = self.pool(input)
        return input



class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        self.conv3_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=5)
            for in_channels in [6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4]
        ])

    def forward(self, input):
        connections = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [0, 1, 3, 5],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [1, 3, 4, 5],
            [0, 1, 4],
            [2, 3, 5],
            [0, 3, 4, 5],
            [1, 2, 4, 5],
            [0, 1, 2, 5],
            [0, 2, 3, 4]
        ]

        outputs = []
        for conv, in_channels in zip(self.conv3_layers, connections):
            sub_x = torch.cat([input[:, i:i+1, :, :] for i in in_channels], dim=1)
            out = conv(sub_x)
            outputs.append(out)

        input = torch.cat(outputs, dim=1)
        input = F.relu(input)
        return input


class S4(nn.Module):
    def __init__(self):
        super(S4, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size = 2, stride =2)

    def forward(self, input):
        input = self.pool(input)
        return input


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

    def forward(self, input):
        input = self.conv5(input)
        input = F.relu(input)
        return input

class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()
        self.fc6 = nn.Linear(120, 84)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        input = self.fc6(input)
        input = F.relu(input)
        return input

class LeNetOutput(nn.Module):
    def __init__(self):
        super(LeNetOutput, self).__init__()
        self.fc7 = nn.Linear(84, 10)

    def forward(self, input):
        input = self.fc7(input)
        return input

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = C1()
        self.s2 = S2()
        self.c3 = C3()
        self.s4 = S4()
        self.c5 = F5()
        self.f6 = F6()
        self.output = LeNetOutput()

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.f6(x)
        x = self.output(x)
        return x

def train():
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f'Batch {i + 1}/{len(trainloader)}, Loss: {loss.item():.4f}')

    return running_loss / len(trainloader)

def test():
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(testloader), accuracy

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    net = LeNet5()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    model_path = './lenet5.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    else:
        print("No saved model found, starting training from scratch")

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train()
        test_loss, accuracy = test()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
