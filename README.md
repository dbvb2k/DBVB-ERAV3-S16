# Pet Image Segmentation with U-Net

A PyTorch implementation of U-Net for pet image segmentation using the Oxford-IIIT Pet Dataset. This project demonstrates various configurations of U-Net architecture for binary segmentation of pet images.

## Overview

This project implements a U-Net architecture for semantic segmentation of pet images, separating the pet from the background. It includes multiple model configurations and data augmentation options to improve segmentation accuracy.

## Features

- Multiple U-Net configurations:
  - MaxPool + Transposed Convolution
  - MaxPool + Bilinear Upsampling
  - Strided Convolution + Transposed Convolution
  - Strided Convolution + Bilinear Upsampling
- Data augmentation options:
  - Random horizontal flips
  - Random rotations
  - Color jittering
  - Random affine transformations
- Training visualization and metrics
- Model performance comparison
- Support for both CPU and GPU training

## Requirements

- Python 3.7+
- PyTorch 2.5+
- torchvision
- PIL
- matplotlib
- numpy
- tqdm

## Dataset

The project uses the Oxford-IIIT Pet Dataset, which should be organized as follows:
data/  
├──oxford_pet/  
├────images/  
├──────image1.jpg  
├──────image2.jpg  
...  
├────annotations/  
├──────trimaps/  
├───────image1.png  
├───────image2.png  
```

Download the dataset from [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
```

## Project Structure
README.md
├── UNet_Model.py       # U-Net model architecture  
├── training.py         # Main training script  
├── results/            # Output directory for visualizations  
└── data/               # Dataset directory  
```  

## Usage & Basic training:

### Without augmentation (default)
python training.py --augment=False

### With augmentation 
python training.py --augment=True



## Model Configurations

The project implements four different U-Net configurations:

1. **Config 1**: MaxPool + TransConv + BCE
   - Uses max pooling for downsampling
   - Uses transposed convolution for upsampling
   - Binary Cross Entropy loss

2. **Config 2**: MaxPool + TransConv + Dice
   - Uses max pooling for downsampling
   - Uses transposed convolution for upsampling
   - Dice loss

3. **Config 3**: StrideConv + TransConv + BCE
   - Uses strided convolution for downsampling
   - Uses transposed convolution for upsampling
   - Binary Cross Entropy loss

4. **Config 4**: StrideConv + Upsample + Dice
   - Uses strided convolution for downsampling
   - Uses bilinear upsampling
   - Dice loss

## Results

The training process generates several visualizations:
- Individual prediction visualizations for each configuration
- Combined comparison visualization
- Training and validation metrics

Results are saved in the `results/` directory. View detailed results and visualizations [here](results.md).

## Model Architecture

The U-Net architecture consists of:
- Encoder path with 4 downsampling steps
- Bottleneck layer
- Decoder path with 4 upsampling steps
- Skip connections between encoder and decoder
- Final 1x1 convolution for binary segmentation

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original U-Net paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Oxford-IIIT Pet Dataset: [Dataset paper](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)

## Contact

For questions or feedback, please open an issue in the repository.