import data
from data import np
from model import torch, optim, nn
import cv2
import model
from modelgraph import create_acc_loss_graph as gr

EPOCHS = 10
BATCH_SIZE = 32
VAL_PCT = 0.2
TRAIN_SIZE = 1000
IMG_SIZE = model.IMG_SIZE

REBUILD_TRAIN_DATA = False
REBUILD_TEST_DATA = False
TRAIN_DATA = True
TEST_DATA = False
SHOW_GRAPH = True


if REBUILD_TRAIN_DATA:
    test = data.SketchData()
    test.build_training_data(TRAIN_SIZE)
if REBUILD_TEST_DATA:
    test = data.SketchData()
    test.build_test_data()

if TRAIN_DATA:
    training_data = np.load(f'{data.SketchData.TRAIN_PATH}/training_data.npy', allow_pickle=True)
    labels = np.load(f'{data.SketchData.TRAIN_PATH}/labels.npy', allow_pickle=True)

    X = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_SIZE, IMG_SIZE) #Input images
    X = (X / 255.0)

    y = torch.Tensor([i[1] for i in training_data]) #Target classes

    val_size = int(len(X) * VAL_PCT)
    print(val_size, 'Val size')

    # Train data
    train_X = X[:-val_size]
    train_y = y[:-val_size]

    # Validation data
    test_X = X[-val_size:]
    test_y = y[-val_size:]
    print(X[0].shape)
    print(len(labels), "Classes")
    net = model.Net(len(labels)).to(model.device)
    print(net)


    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    #Train model
    model.train(net, BATCH_SIZE, EPOCHS, train_X, train_y, loss_function, optimizer, True, test_X, test_y)
    accuracy, validation_loss = model.validate(net, test_X, test_y, loss_function, BATCH_SIZE) #Validation
    torch.save(net.state_dict(), 'mytraining.pt') #Save model
    print(f'Validation Accuracy {accuracy}, Validation Loss {validation_loss}') #Accuracy

    """
    cv2.imshow(f'{training_data[0][1]}', training_data[0][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
if TEST_DATA:

    test_data = np.load(f'{data.SketchData.TEST_PATH}/test_data.npy', allow_pickle=True)
    labels = np.load(f'{data.SketchData.TRAIN_PATH}/labels.npy', allow_pickle=True)
    X_Test = torch.tensor([i[0] for i in test_data]).view(-1, IMG_SIZE, IMG_SIZE)
    print("argmax", torch.argmax(X_Test))
    # X_Test = X_Test / 255.0
    print(X_Test[0][0])
    name_Test = [i[1] for i in test_data]

    try:
        net
        print("Net exists")
    except NameError:
        print("Loading NN")
        net = model.Net(len(labels)).to(model.device)
        net.load_state_dict(torch.load('mytraining.pt'))
        net.eval()

    print(net)
    print("-----default model----")
    def_pred  = model.test(net, X_Test.float(), name_Test, labels)
    print("---------model1------")
    net.load_state_dict(torch.load('model1.pt'))
    net.eval()
    mod1_pred = model.test(net, X_Test.float(), name_Test, labels)
    print("------model2------")
    net.load_state_dict(torch.load('model2.pt'))
    net.eval()
    mod2_pred = model.test(net, X_Test.float(), name_Test, labels)

    results = [[x, y, z] for x, y, z in zip(def_pred, mod1_pred, mod2_pred)]
    for r, name in zip(results, name_Test):
        print(f'{name} is {np.unique(r)}')
if SHOW_GRAPH:
    gr(model.MODEL_NAME, EPOCHS, BATCH_SIZE, TRAIN_SIZE)
