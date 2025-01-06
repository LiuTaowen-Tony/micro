import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from ml_utils.misc import to_model_device
from model.clip import Clip
from torch import FloatTensor
from typing import List, Union
from ml_utils.visualize import fig_to_array


class ClipInference:
    def __init__(self, model: Clip, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def confusion_matrix(
        self, images: FloatTensor, texts: Union[str, List[str], torch.LongTensor]
    ):
        """
        Computes the confusion matrix for the given images and texts.
        Args:
            images (torch.Tensor): A tensor containing the images.
            texts (Union[str, List[str], torch.Tensor]): The texts to be tokenized and used for inference.
            Can be a single string, a list of strings, or a pre-tokenized tensor.
        Returns:
            torch.Tensor: The computed logits representing the confusion matrix.
        """
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            input_ids = to_model_device(input_ids, self.model)
            attention_mask = to_model_device(attention_mask, self.model)
            text, img = self.model.forward(input_ids, images, attention_mask)
            text = F.normalize(text, p=2, dim=-1)
            img = F.normalize(img, p=2, dim=-1)
            logits = text @ img.t()
            return logits

    def find_matches(self, images, queries, top_k=9):
        logits = self.confusion_matrix(images, queries)
        values, indices = logits.topk(top_k)
        return values, indices

    def show_confusion_matrix(self, images, texts, cutoff=10):
        """
        Create a confusion matrix visualization with texts on y-axis and image numbers on x-axis.
        
        Args:
            images: Tensor of images (N, C, H, W)
            texts: List of text strings
            cutoff: Maximum number of images/texts to display
            
        Returns:
            numpy.ndarray: The visualization as a numpy array
        """
        # Limit the number of images and texts
        images = images[:cutoff]
        texts = texts[:cutoff]
        
        # Calculate confusion matrix logits
        logits = self.confusion_matrix(images, texts)
        matrix_data = logits.float().cpu().numpy()
        
        # Create figure with larger size to accommodate images
        fig = plt.figure(figsize=(20, 14))
        
        # Calculate grid layout for images
        n_images = len(images)
        n_cols = min(5, n_images)  # Max 5 images per row
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create GridSpec with proper layout and spacing
        total_rows = n_rows + 1
        gs = plt.GridSpec(
            total_rows, 
            n_cols, 
            height_ratios=[1.5] + [1] * (total_rows - 1),
            hspace=0.5,
            wspace=0.3
        )
        
        # Upper subplot for confusion matrix (spans all columns)
        ax_matrix = plt.subplot(gs[0, :])
        im = ax_matrix.imshow(matrix_data, cmap='viridis')
        
        # Add colorbar with proper sizing
        cbar = plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
        
        # Add text labels with adjusted fontsize
        for i in range(len(texts)):
            for j in range(len(images)):
                text = f'{matrix_data[i, j]:.2f}'
                ax_matrix.text(j, i, text, ha='center', va='center', fontsize=8)
        
        # Set labels and title for matrix with adjusted spacing
        ax_matrix.set_xticks(range(len(images)))
        ax_matrix.set_yticks(range(len(texts)))
        ax_matrix.set_xticklabels([f'Image {i}' for i in range(len(images))], rotation=45, ha='right', rotation_mode='anchor')
        ax_matrix.set_yticklabels(texts)
        
        # Adjust title position
        ax_matrix.set_title('Confusion Matrix', pad=20)
        
        # Plot images in remaining grid spots
        for idx, img in enumerate(images):
            row = idx // n_cols + 1  # +1 because matrix is in row 0
            col = idx % n_cols
            ax = plt.subplot(gs[row, col])
            
            img_np = img.permute(1, 2, 0).cpu().numpy()
            
            # Normalize image
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
            ax.imshow(img_np)
            ax.axis('off')
            ax.set_title(f'Image {idx}', pad=10)
        
        # Adjust layout to prevent overlapping
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.95)
        
        # Convert to numpy array and clean up
        array = fig_to_array(fig)
        plt.close(fig)
        return array