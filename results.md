# Results and Visualizations

This page contains the results and visualizations from training different U-Net configurations on the Oxford-IIIT Pet Dataset.

## Training Details

- Number of epochs: 20
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Image size: 128x128
- Dataset split: 80% training, 20% validation

## Training logs, Individual Model ooutputs and Model Performance Comparison
### Training Log-1
<img src="results/train_scr1.png" width="1000"/>

### Training Log-2
<img src="results/train_scr2.png" width="1000"/>

## Model Performance Comparison
<img src="results/all_configs_comparison.png" width="1000"/>

## Individual Configuration Results

### Configuration 1: MaxPool + TransConv + BCE
<img src="results/predictions_config_1.png" width="500"/>

### Configuration 2: MaxPool + TransConv + Dice
<img src="results/predictions_config_2.png" width="500"/>

### Configuration 3: StrideConv + TransConv + BCE
<img src="results/predictions_config_3.png" width="500"/>

### Configuration 4: StrideConv + Upsample + Dice
<img src="results/predictions_config_4.png" width="500"/>




