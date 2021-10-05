import torch

import cargan


class PitchDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Layer parameters
        in_channels = [1, 1024, 128, 128, 128, 256]
        out_channels = [1024, 128, 128, 128, 256, 512]
        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        pads = [254] + 5 * [32]

        # Create model
        model = []
        iterator = zip(in_channels, out_channels, kernel_sizes, strides, pads)
        for item in iterator:
            model.extend([
                torch.nn.utils.weight_norm(torch.nn.Conv1d(*item)),
                torch.nn.LeakyReLU(.2, True),
                torch.nn.MaxPool1d(2, 2)])
        self.model = torch.nn.Sequential(*model)
        self.classifier = torch.nn.Linear(2048, cargan.PITCH_BINS)
    
    def forward(self, x):
        # shape=(batch * samples / 1024, 1, 1024)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0] * x.shape[1] // 1024, 1024, 1)
        x = x.permute(0, 2, 1)

        # Forward pass
        x  = self.model(x)

        # Reshape
        x = x.permute(0, 2, 1).reshape(-1, 2048)

        # Compute logits
        return self.classifier(x)
