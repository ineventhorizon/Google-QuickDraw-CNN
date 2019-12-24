import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import tqdm
import time
MODEL_NAME = f"model-{int(time.time())}"


#Expected img size
IMG_SIZE = 100

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    #print(device)
    print(f'Running on {torch.cuda.get_device_name(device)}')
    print(f'CUDA Device count: {torch.cuda.device_count()}')

class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 Input, 32 Convolutional Feature, 3 Kernel Size
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 Input, 64 Convolutional Feature, 3 Kernel Size
        self.conv3 = nn.Conv2d(64, 128, 3)  # 64 Input, 128 Convolutional Feature, 3 Kernel Size
        self.fc1 = nn.Linear(12800 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        # First Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # Second Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # Third Conv layer, Conv layer->Relu activation-> (2,2) max pooling
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # print(torch.flatten(x, 1).shape)
        # To pass x to fully connected layer we need to apply flatten to x
        x = torch.flatten(x, 1)
        # First fully connected layer, Fully connected layer -> relu activation function
        x = self.fc1(x)
        x = F.relu(x)
        # Second fully connected layer, Fully connected layer -> sigmoid activation (output)
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x #Return output



#Training function
def train(net, BATCH, EPOCH, trainX, target, loss_function, optimizer, validateFlag=False, validateX=None, validateY=None):
    print(f'Training on {device}')
    BATCH_SIZE = BATCH
    EPOCHS = EPOCH
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            net.train()
            for i in tqdm(range(0, len(trainX), BATCH_SIZE)):
                batch_X = trainX[i:i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device)
                batch_y = target[i:i + BATCH_SIZE].to(device)

                net.zero_grad()
                outputs = net(batch_X)
                matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
                acc = matches.count(True) / len(matches)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()

                if validateFlag and i % 200 == 0:
                    val_acc, val_loss = validate(net, validateX, validateY, loss_function, BATCH_SIZE)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},in_sample,{round(float(acc), 2)},{round(float(loss), 4)}"
                            f",validation,{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")
                #print(f'Accuracy: {val_acc}')
            print(f'Epoch: {epoch}, Loss : {loss}')

#Validation function
def validate(net, test_X, test_y, loss_function, batch):
    print(f'Testing on {device}')
    net.eval()
    BATCH_SIZE = batch
    correct = 0
    total = 0
    accuracies = []
    losses = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_X), BATCH_SIZE)):
            #real_class = torch.argmax(test_y[i]).to(device)


            batch_y = test_y[i:i+BATCH_SIZE].to(device)
            net_out = net(test_X[i:i+BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device))
            #print(net_out)
            matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(net_out, batch_y)]
            acc = matches.count(True) / len(matches)
            loss = loss_function(net_out, batch_y)
            #predicted_class = torch.argmax(net_out)

            accuracies.append(float(acc))
            losses.append(float(loss))



            """
            if predicted_class == real_class:
                correct += 1
            total += 1
            """

    #print(f'Accuracy {round(correct/total, 3)}')
    return sum(accuracies) / len(accuracies), sum(losses) / len(losses)

#Test function
def test(net, X_Test, Name_Test, labels):
    pred = []
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(len(X_Test))):
            #print(X_Test[i].view(-1, 1, 28, 28).to(device).shape)
            out = net(X_Test[i].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device))[0]
            predicted_class = torch.argmax(out)
            #print(f'{Name_Test[i]} is {labels[predicted_class][0]}')
            pred.append(labels[predicted_class][0])
            #print(out)
    return pred
