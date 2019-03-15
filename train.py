# coding: utf-8

import torch
import torch.nn as nn

from util import load_data
from model import LinerRegression

if  __name__ == "__main__":

    X, Y = load_data()

    model = LinerRegression(X.shape[1])

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.6)

    for epoch in range(1000):
        inputs = torch.from_numpy(X)
        targets = torch.from_numpy(Y)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))

    y_pred = model(torch.from_numpy(X)).data.numpy()

    print(y_pred[0:5])
    print(Y[0:5])
