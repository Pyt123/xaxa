# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import content
import time
import torch
import pickle as pkl
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

GLOBAL_COUNTER = 1

def save_model_as_numpy(model):
    i = 1
    for parameter in model.parameters():
        nump = parameter.cpu().type(torch.FloatTensor).data.numpy().astype(np.float16)
        if i % 2 == 1:
            nump = np.transpose(nump)
        np.save('model/params' + str(i), nump, allow_pickle=True)
        i += 1


def load_model_from_file():
    return (np.load('model/params1.npy'), np.load('model/params2.npy'),
            np.load('model/params3.npy'), np.load('model/params4.npy'),
            np.load('model/params5.npy'), np.load('model/params6.npy'),
            np.load('model/params7.npy'), np.load('model/params8.npy'),
            np.load('model/params9.npy'), np.load('model/params10.npy'),
            np.load('model/params11.npy'), np.load('model/params12.npy'))


def save_model_as_txt(par1, par2, par3, par4):
        np.savetxt('model/params1.txt', par1)
        np.savetxt('model/params2.txt', par2)
        np.savetxt('model/params3.txt', par3)
        np.savetxt('model/params4.txt', par4)


def relu(x):
    return x * (x > 0)


def get_predicted(x, model):
    save_model_as_numpy(model)
    (w1, p1, w2, p2, w3, p3, w4, p4, w5, p5, w6, p6) = load_model_from_file()

    output_array = []
    layer1 = np.empty((336,), order='C')
    layer2 = np.empty((336,), order='C')
    layer3 = np.empty((336,), order='C')
    layer4 = np.empty((336,), order='C')
    layer5 = np.empty((336,), order='C')
    layer6 = np.empty((36,), order='C')

    length = len(x)
    for i in range(0, length):
        np.matmul(x[i], w1, out=layer1)
        layer1 = relu(layer1 + p1)

        np.matmul(layer1, w2, out=layer2)
        layer2 = relu(layer2 + p2)

        np.matmul(layer2, w3, out=layer3)
        layer3 = relu(layer3 + p1)

        np.matmul(layer3, w4, out=layer4)
        layer4 = relu(layer4 + p4)

        np.matmul(layer4, w5, out=layer5)
        layer5 = relu(layer5 + p5)

        np.matmul(layer5, w6, out=layer6)
        layer6 = relu(layer6 + p6)

        output_array.append(layer6.argmax())

    output_vector = np.array(output_array)
    return output_vector.reshape((len(output_vector), 1))


EPOCHS = 1000


def train(x_train, y_train, num_of_try, learning_rate, epsilon):
    #START_MOMENTUM = 1
    #MOMENTUM = START_MOMENTUM
    #START_LR = 1
    #LR = START_LR
    #DIVIDER_MOM = 1.03
    #DIVIDER_LR = 1.05
    EPOCHS_TO_CHANGE = 30
    NEXT_TO_CHANGE = EPOCHS_TO_CHANGE
    LAST_BEST = 0
    BEST_EPOCH = 0

    # Create model
    #model = content.NeuralNet().cuda()
    # Load model
    model = torch.load('mytraining.pth')

    # Some stuff
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Convert numpy arrays to torch variables
    inputs = torch.autograd.Variable(torch.from_numpy(x_train).type(torch.cuda.FloatTensor), requires_grad=True)
    targets = torch.autograd.Variable(torch.from_numpy(y_train).type(torch.cuda.LongTensor), requires_grad=False)
    targets = targets.squeeze(1)

    for epoch in range(EPOCHS):
        if epoch == NEXT_TO_CHANGE:
            #MOMENTUM /= DIVIDER_MOM
            #LR /= DIVIDER_LR
            #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
            NEXT_TO_CHANGE += EPOCHS_TO_CHANGE
            ratio = test(model, y_val)
            if ratio > LAST_BEST:
                torch.save(model, 'bestmodel' + str(num_of_try) + '.pth')
                LAST_BEST = ratio
                BEST_EPOCH = epoch
                print('\n-----Best epoch: ' + str(BEST_EPOCH) + '\tBest ratio: ' + str(LAST_BEST) + '-----\n')

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % EPOCHS_TO_CHANGE == 0:
            print('Epoch [{}/{}],\tLoss: {:.24f}'.format(epoch, EPOCHS, loss.data[0]))

    torch.save(model, 'endingmodel' + str(num_of_try) + '.pth')
    ratio = test(model, y_val)
    if ratio > LAST_BEST:
        LAST_BEST = ratio
        BEST_EPOCH = epoch
        torch.save(model, 'bestmodel' + str(num_of_try) + '.pth')

    print('END-----Best epoch: ' + str(BEST_EPOCH) + '\tBest ratio: ' + str(LAST_BEST) + '-----\n')
    return model


def test(model, y_val):
    pred = get_predicted(x_val, model)
    good = 0
    for i in range(0, 2634):
        if pred[i] == y_val[i]:
            good += 1
    ratio = (good / 2634.0) * 100
    print("ratio: " + str(ratio))
    return ratio


(x, y) = pkl.load(open('train.pkl', mode='rb'))
(x_train, y_train) = (x[:27500], y[:27500])
(x_val, y_val) = (x[27500:], y[27500:])

INCREASE_EPOCHS = 2000

learning_rates = [0.00006]
epsilons = [0.001]
for i in range(len(learning_rates)):
    for j in range(len(epsilons)):
        print('Testing learning rate = ' + str(learning_rates[i]) + ' and epsilon = ' + str(epsilons[j]))
        train(x_train, y_train, i + 100 * j, learning_rates[i], epsilons[j])
        print('\n')
    print('\n')
    EPOCHS += INCREASE_EPOCHS

exit(0)

#save_model_as_numpy(model)
#print('saved as numpy')
