import torch

import cargan


class Autoregressive(torch.nn.Module):

    def __init__(self):
        super().__init__()

        model = [
            torch.nn.Linear(cargan.AR_INPUT_SIZE, cargan.AR_HIDDEN_SIZE),
            torch.nn.LeakyReLU(.1)]
        for _ in range(3):
            model.extend([
                torch.nn.Linear(
                    cargan.AR_HIDDEN_SIZE,
                    cargan.AR_HIDDEN_SIZE),
                torch.nn.LeakyReLU(.1)])
        model.append(
            torch.nn.Linear(cargan.AR_HIDDEN_SIZE, cargan.AR_OUTPUT_SIZE))
        self.model = torch.nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x.squeeze(1))
