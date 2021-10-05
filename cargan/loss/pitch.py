import torch
import torchcrepe


###############################################################################
# CREPE perceptual loss
###############################################################################


class CREPEPerceptualLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        # Register model
        self.add_module('model', torchcrepe.Crepe())

        # Don't update model weights
        self.requires_grad_(False)
    
    def forward(self, x, y):
        # Get feature maps
        x_maps = self.activations(x)
        y_maps = self.activations(y)

        # Compute distance
        loss = 0.
        for x_map, y_map in zip(x_maps, y_maps):
            loss += torch.nn.functional.l1_loss(x_map, y_map)
        
        return loss

    def activations(self, x):
        activations = []
        
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through model and save activations
        x = self.model.layer(x, self.model.conv1, self.model.conv1_BN, (0, 0, 254, 254))
        activations.append(x)
        x = self.model.layer(x, self.model.conv2, self.model.conv2_BN)
        activations.append(x)
        x = self.model.layer(x, self.model.conv3, self.model.conv3_BN)
        activations.append(x)
        x = self.model.layer(x, self.model.conv4, self.model.conv4_BN)
        activations.append(x)
        x = self.model.layer(x, self.model.conv5, self.model.conv5_BN)
        activations.append(x)
        x = self.model.layer(x, self.model.conv6, self.model.conv6_BN)
        activations.append(x)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.model.in_features)

        # Compute unnormalized probability distribution
        x = self.model.classifier(x)
        activations.append(x)

        return activations
