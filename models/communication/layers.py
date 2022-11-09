import torch.nn as nn


def mlp_encoder(units: list):
    layers = []
    prev_layer_size = units[0]
    hidden_units = units[1:]
    for size in hidden_units:
        layers.append(nn.Linear(prev_layer_size, size))
        layers.append(nn.ReLU())
        prev_layer_size = size
    return nn.Sequential(*layers)


def mlp_decoder(units: list):
    layers = []
    prev_layer_size = units[0]
    hidden_units = units[1:]
    for i, size in enumerate(hidden_units):
        layers.append(nn.Linear(prev_layer_size, size))
        if i < len(hidden_units) - 1:
            layers.append(nn.ReLU())
        else:
            nn.init.kaiming_uniform_(layers[-1].weight)
            nn.init.constant_(layers[-1].bias, 0)
        prev_layer_size = size
    return nn.Sequential(*layers)
