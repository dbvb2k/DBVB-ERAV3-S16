# Results and Visualizations

This page contains the results and visualizations from training different U-Net configurations on the Oxford-IIIT Pet Dataset.

## Training Details

- Number of epochs: 20
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Image size: 128x128
- Dataset split: 80% training, 20% validation

## Training logs
![Training Log-1](results/predictions_config_1.png)
![Training Log-2](results/predictions_config_1.png)

## Model Performance Comparison

![All Configurations Comparison](results/all_configs_comparison.png)

## Individual Configuration Results

### Configuration 1: MaxPool + TransConv + BCE
![Config 1 Results](results/predictions_config_1.png)

### Configuration 2: MaxPool + TransConv + Dice
![Config 2 Results](results/predictions_config_2.png)

### Configuration 3: StrideConv + TransConv + BCE
![Config 3 Results](results/predictions_config_3.png)

### Configuration 4: StrideConv + Upsample + Dice
![Config 4 Results](results/predictions_config_4.png)

## Performance Metrics

| Configuration | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|--------------|---------------|-----------------|-------------------|-------------------|
| Config 1     | TBD           | TBD             | TBD               | TBD               |
| Config 2     | TBD           | TBD             | TBD               | TBD               |
| Config 3     | TBD           | TBD             | TBD               | TBD               |
| Config 4     | TBD           | TBD             | TBD               | TBD               |



