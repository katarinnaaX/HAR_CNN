# Human Action Recognition using CNN (MobileNetV2)

10-class image classification of human actions using transfer learning 
with MobileNetV2, implemented in Python/Keras.

## Dataset
- Human Action Recognition dataset - 8,406 images across 10 classes
- Classes: clapping, cycling, dancing, eating, fighting, hugging, 
  laughing, running, sleeping, using_laptop
- Train/val/test split: 6400 / 1600 / 406
  
## Pipeline
1. **Preprocessing** - images resized to 128×128; pixel values normalized to [−1, 1] via a rescaling layer to match MobileNetV2 input expectations
2. **Data augmentation** - horizontal flip, rotation (±18°), zoom (5%), contrast (10%); deliberately moderate to avoid degrading training accuracy
3. **Model**
- Base: MobileNetV2 pretrained on ImageNet (frozen)
- Head: GlobalAveragePooling2D → Dense(64) → Dense(32) → Dense(10)
- Regularization: Dropout(0.3), data augmentation, early stopping
4. **Training** - 2-phase: FC-only then full fine-tuning with BatchNorm layers frozen

## Results
| Split      | Accuracy |
|------------|----------|
| Train      | 87.86%   |
| Validation | 80.56%   |
| Test       | 80.79%   |

Best classes: eating (F1=0.99), cycling (F1=0.98)  
Hardest: clapping (F1=0.70) - visually similar to hugging/dancing

## Files
- `nm.ipynb` - full training notebook
- `nm.pdf` - detailed project report (Serbian)

## Requirements
tensorflow, numpy, matplotlib, scikit-learn
