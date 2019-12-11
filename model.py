import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import tqdm


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    #print(device)
    print(f'Running on {torch.cuda.get_device_name(device)}')
    print(f'CUDA Device count: {torch.cuda.device_count()}')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 Input, 32 Convolutional Feature, 5 Kernel Size
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 Input, 64 Convolutional Feature, 5 Kernel Size
        self.conv3 = nn.Conv2d(64, 128, 3)  # 64 Input, 128 Convolutional Feature, 5 Kernel Size
        self.fc1 = nn.Linear(128*1*1, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        #First Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        #Second Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        #Third Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #print(torch.flatten(x, 1).shape)
        #To pass x to fully connected layer we need to apply flatten to x
        x = torch.flatten(x, 1)
        #First fully connected layer, Fully connected layer -> relu activation function
        x = self.fc1(x)
        x = F.relu(x)
        # Second fully connected layer, Fully connected layer -> sigmoid activation (output)
        x = self.fc2(x)
        x = F.softmax(x, 1)
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        #For finding fc1 input size
        #print(x.shape)
        #print(x[0].shape)
        return x

def train(net, BATCH, EPOCH, trainX, target, loss_function, optimizer):
    BATCH_SIZE = BATCH
    EPOCHS = EPOCH

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(trainX), BATCH_SIZE)):
            batch_X = trainX[i:i + BATCH_SIZE].view(-1, 1, 28, 28).to(device)
            batch_y = target[i:i + BATCH_SIZE].to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(loss)

def validate(net,test_X, test_y):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 28, 28).to(device))[0]
            #print(net_out)
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print(f'Accuracy {round(correct/total, 3)}')
    return correct/total
def test(net, X_Test, Y_Test):
    with torch.no_grad():
        for i in tqdm(range(len(X_Test))):
            out = net(X_Test[i].view(-1, 1, 28, 28).to(device))[0]
            predicted_class = torch.argmax(out)
            print(f'{Y_Test[i]} is {predicted_class}')
            print(out)

