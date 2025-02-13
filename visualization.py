import matplotlib.pyplot as plt
import torch
import os

def visualize_predictions(model, val_loader, device, num_images=4, metrics=None):
    """
    Visualize predictions from the model compared to ground truth
    Args:
        model: trained UNet model
        val_loader: validation data loader
        device: torch device
        num_images: number of images to visualize
        metrics: dictionary containing model metrics (acc, loss, val_acc, val_loss)
    """
    model.eval()
    
    # Get a batch of images
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = torch.sigmoid(model(images))
        predictions = (predictions > 0.5).float()
    
    # Move tensors to CPU for plotting
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    # Create figure with extra space for metrics and titles
    fig = plt.figure(figsize=(15, 5*num_images + 3))  # Increased height for better spacing
    
    # Create grid spec to handle both images and metrics text
    gs = fig.add_gridspec(num_images + 1, 3, height_ratios=[1]*num_images + [0.3])
    
    # Add title with more space
    plt.suptitle('Image, Ground Truth Mask, and Predicted Mask Comparison', 
                 fontsize=16, 
                 y=0.98)  # Keep title at top
    
    # Adjust subplot spacing to leave more room for titles
    plt.subplots_adjust(
        top=0.92,     # Reduced from 0.95 to leave more space at top
        bottom=0.08,  # Adjusted bottom margin
        hspace=0.4,   # Increased spacing between rows
        wspace=0.3    # Added spacing between columns
    )
    
    for idx in range(min(num_images, len(images))):
        # Create subplot for each row
        ax1 = fig.add_subplot(gs[idx, 0])
        ax2 = fig.add_subplot(gs[idx, 1])
        ax3 = fig.add_subplot(gs[idx, 2])
        
        # Original image - normalize to [0,1] range for display
        img = images[idx].permute(1, 2, 0)  # Change from CxHxW to HxWxC
        img = torch.clamp(img, 0, 1)  # Ensure values are in [0,1]
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Ground truth mask
        mask = masks[idx].squeeze()
        ax2.imshow(mask.numpy(), cmap='gray')
        ax2.set_title('Ground Truth Mask')
        ax2.axis('off')
        
        # Predicted mask
        pred = predictions[idx].squeeze()
        ax3.imshow(pred.numpy(), cmap='gray')
        ax3.set_title('Predicted Mask')
        ax3.axis('off')
    
    # Add metrics text at the bottom if provided
    if metrics is not None:
        metrics_text = (
            f"Training Metrics:\n"
            f"Loss: {metrics['loss']:.4f} | Accuracy: {metrics['acc']:.4f}\n"
            f"Validation Metrics:\n"
            f"Loss: {metrics['val_loss']:.4f} | Accuracy: {metrics['val_acc']:.4f}"
        )
        
        # Add text in the bottom subplot
        ax_metrics = fig.add_subplot(gs[-1, :])
        ax_metrics.text(0.5, 0.5, metrics_text,
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax_metrics.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                       fontsize=20)
        ax_metrics.axis('off')
    
    plt.tight_layout()
    return fig

def compare_all_configs(configs, num_configs=4):
    """
    Display results from all configurations side by side
    Args:
        configs: list of configuration tuples
        num_configs: number of configurations to display
    """
    try:
        # Define colors for each configuration
        config_colors = {
            1: 'tab:blue',      # Blue
            2: 'tab:green',     # Green
            3: 'tab:orange',    # Orange
            4: 'tab:red'        # Red
        }
        
        # Create a figure with more height to accommodate titles and legend
        fig = plt.figure(figsize=(20, 18))
        
        # Add main title with more top margin
        plt.suptitle('Comparison of All Configurations', 
                    fontsize=20, 
                    y=0.98)  # Move title higher
        
        # Adjust the subplot layout - leave more space at top for title and legend
        plt.subplots_adjust(top=0.85,    # Reduced to make room for legend
                           bottom=0.05,
                           hspace=0.25)
        
        for config_idx in range(1, num_configs + 1):
            img_path = f'results/predictions_config_{config_idx}.png'
            
            if not os.path.exists(img_path):
                print(f"Warning: Could not find results for configuration {config_idx}")
                continue
                
            # Read and display the image
            img = plt.imread(img_path)
            ax = fig.add_subplot(2, 2, config_idx)
            ax.imshow(img)
            
            # Get configuration name
            config_name = configs[config_idx-1][3]
            
            # Add colored title with more padding
            ax.set_title(f'Configuration {config_idx}: {config_name}', 
                        pad=20,
                        fontsize=12,
                        wrap=True,
                        color=config_colors[config_idx],  # Add color
                        fontweight='bold')               # Make it bold
            ax.axis('off')
        
        # Add a legend below the title, center-justified
        legend_elements = [plt.Line2D([0], [0], color=color, lw=4, 
                                    label=f'Config {idx}: {configs[idx-1][3]}')
                         for idx, color in config_colors.items()]
        fig.legend(handles=legend_elements, 
                  loc='upper center',       # Center
                  bbox_to_anchor=(0.5, 0.95),  # Center horizontally
                  ncol=2,
                  fontsize=10)
        
        # Save the comparison figure
        comparison_path = 'results/all_configs_comparison.png'
        plt.savefig(comparison_path, 
                   bbox_inches='tight', 
                   pad_inches=1.0,
                   dpi=300)
        plt.close()
        
        print(f"\nSaved configuration comparison to {comparison_path}")
        
    except Exception as e:
        print(f"Error comparing configurations: {str(e)}")
        raise 