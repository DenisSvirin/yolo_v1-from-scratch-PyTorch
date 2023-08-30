import numpy as np
import torch

def train_epoch(model, optimizer, criterion):
    loss_log = []
    for x_batch, y_batch in train_loder:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        pred = model(x_batch)

        loss = criterion(pred, y_batch)

        loss_log.append(loss.item())

        loss.backward()

        optimizer.step()

    return loss_log

@torch.no_grad()
def test_epoch(model, criterion):
    loss_log = []
    for x_batch, y_batch in test_loder:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)

        loss = criterion(pred, y_batch)

        loss_log.append(loss.item())

    return loss_log

def train(model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss = train_epoch(model, optimizer, criterion)
        test_loss = test_epoch(model, criterion)

        print("epoch: ", epoch, " | ",
         "train loss: ", np.mean(train_loss), " | ",
         "test loss: ", np.mean(test_loss))

if __name__ == "__main__":
