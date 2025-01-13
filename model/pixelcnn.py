import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MaskedConv2d(nn.Conv2d):
    """
    A 2D convolution with a mask to enforce autoregressive property.
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type="A", stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.register_buffer("mask", self.create_mask(kernel_size, mask_type))

    def create_mask(self, kernel_size, mask_type):
        """
        Creates the mask to enforce autoregressive constraints.
        
        Args:
        - kernel_size (int or tuple): Size of the convolution kernel.
        - mask_type (str): "A" or "B" (Type A excludes the center pixel, Type B includes it).
        
        Returns:
        - torch.Tensor: Mask tensor.
        """
        height, width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        mask = torch.ones(1, 1, height, width)

        # Block future positions (below and to the right)
        mask[:, :, height // 2, width // 2 + (mask_type == "B"):] = 0
        mask[:, :, height // 2 + 1:] = 0

        return mask

    def forward(self, x):
        self.weight.data *= self.mask  # Apply the mask to the weights
        return super().forward(x)


class PixelCNN(nn.Module):
    """
    A PixelCNN model for image autoregressive generation.
    """
    def __init__(self, input_channels, num_filters, kernel_size, num_layers, output_channels):
        super(PixelCNN, self).__init__()
        layers = []

        # First layer: Mask Type "A"
        layers.append(MaskedConv2d(input_channels, num_filters, kernel_size, mask_type="A", padding=kernel_size // 2))
        layers.append(nn.ReLU())

        # Intermediate layers: Mask Type "B"
        for _ in range(num_layers - 1):
            layers.append(MaskedConv2d(num_filters, num_filters, kernel_size, mask_type="B", padding=kernel_size // 2))
            layers.append(nn.ReLU())

        # Final layer: Predict output
        layers.append(nn.Conv2d(num_filters, input_channels * output_channels, kernel_size=1))
        self.network = nn.Sequential(*layers)
        self.output_channels = output_channels

    def forward(self, x):
        bsz, channels, height, width = x.size()
        result = self.network(x).view(bsz, channels, self.output_channels, height, width)
        result = result.permute(0, 1, 3, 4, 2).contiguous().view(bsz, channels, height, width, self.output_channels)
        return result


def generate_image(model, img_size, device, num_channels=1):
    """
    Generates an image pixel-by-pixel using an autoregressive PixelCNN model.
    
    Args:
    - model (nn.Module): The trained PixelCNN model.
    - img_size (tuple): Size of the image (height, width).
    - device (torch.device): Device for computation.
    - num_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
    
    Returns:
    - torch.Tensor: Generated image tensor of shape (1, num_channels, height, width).
    """
    model.eval()
    height, width = img_size
    img = torch.zeros(1, num_channels, height, width).to(device)  # Start with a blank image

    with torch.no_grad():
        for i in range(height):
            for j in range(width):
                for c in range(num_channels):
                    # Predict pixel distribution at (i, j, c)
                    output = model(img)
                    probs = F.softmax(output[:, :, i, j], dim=-1).squeeze()  # Pixel probabilities
                    pixel_value = torch.multinomial(probs, num_samples=1)  # Sample a value
                    img[0, c, i, j] = pixel_value / 255.0  # Normalize to [0, 1]

    return img


if __name__ == "__main__":
    # Hyperparameters
    input_channels = 1  # Grayscale
    num_filters = 64
    kernel_size = 7
    num_layers = 5
    output_channels = 256  # 256 discrete pixel intensity levels
    img_size = (28, 28)  # Small image size for quick generation

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN(input_channels, num_filters, kernel_size, num_layers, output_channels).to(device)

    # Generate an image
    generated_img = generate_image(model, img_size, device).cpu().numpy()

    # Visualize the generated image
    plt.imshow(generated_img.squeeze(), cmap="gray")
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()
