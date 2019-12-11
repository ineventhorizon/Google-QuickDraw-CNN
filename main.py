import data
from data import np
from model import torch, optim, nn
import cv2
import model




REBUILD_TRAIN_DATA = False
REBUILD_TEST_DATA = True
TRAIN_DATA = True

if REBUILD_TRAIN_DATA:
    test = data.SketchData()
    test.build_training_data()
if REBUILD_TEST_DATA:
    test = data.SketchData()
    test.build_test_data()

if TRAIN_DATA:
    training_data = np.load(f'{data.SketchData.TRAIN_PATH}/training_data.npy', allow_pickle=True)
    test_data = np.load(f'{data.SketchData.TEST_PATH}/test_data.npy', allow_pickle=True)

    VAL_PCT = 0.1
    X = torch.Tensor([i[0] for i in training_data]).view(-1, 28, 28)
    X = X/255.0

    y = torch.Tensor([i[1] for i in training_data])

    val_size = int(len(X) * VAL_PCT)
    print(val_size, 'Val size')

    # Train data
    train_X = X[:-val_size]
    train_y = y[:-val_size]

    # Validation data
    test_X = X[-val_size:]
    test_y = y[-val_size:]
    print(X[0].shape)

    net = model.Net().to(model.device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    model.train(net, 100, 1, train_X, train_y, loss_function, optimizer)
    model.validate(net, test_X, test_y)

    X_Test = torch.tensor([i[0] for i in test_data]).view(-1, 28, 28).to(model.device)
    #X_Test = X_Test / 255.0
    y_Test = [i[1] for i in test_data]

    model.test(net,X_Test.float(), y_Test)







    #output = net(X[0].view(-1, 1, 28, 28))
    #print(training_data[0])
    #print(output)

    cv2.imshow(f'{training_data[0][1]}', training_data[0][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

