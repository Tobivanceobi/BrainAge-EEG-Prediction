import torch.cuda
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from DataLoader.dataObject import DataObject
from nnModel import ModelV1


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

dl = DataObject(data_cache_id="test", subject_count=500, state="EC", load_cache=True, crop_start=5, crop_end=15)
dl.load_eeg_data()
X_train = dl.x_train
X_target = dl.x_target

X_train = X_train.to(device)
X_target = X_target.to(device)

model = ModelV1()
model.to(device)
# Setup a loss function
loss_fn = nn.CTCLoss()

# Setup an optimizer
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.0001)

epochs = 50

for epoch in range(0, epochs):
    # Set model to training mode
    model.train()

    for sub in range(0, len(X_train)):

        x_data = X_train[sub].unsqueeze(dim=0)

        # Forward pass
        y_pred = model(x_data)
        # Calculate the loss
        loss = loss_fn(y_pred, X_target[sub], 1, 1)

        # Optimizer zero grad (zero optimizer for each iteration)
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step (perform gradient decent)
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")


