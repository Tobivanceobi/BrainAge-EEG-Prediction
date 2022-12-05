import torch
from torch import nn

from nnModel.model_v1 import ModelV1


def info():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = ModelV1().to(device)
    print(model)
    input_image = torch.rand(3, 28, 28)
    print(input_image.size())
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)
    print(logits.size())
