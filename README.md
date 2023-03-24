# Abstract to Title Generation using Sequence2Sequence Models 
Here we are using `model-t51-base` sequenec2sequence model for generation 
of Paper title from Abstract. The task belong to Text Summarization domain.


## Requirements
```
pip install -r requirements.txt
```

## Usage 
### For Training Please follow `training_and_dataprepration.ipynb`
### Processing data
 ```
 python data_processing.py
 ```
 ### Training 
 ```
 python training.py
 ```
### Inference
```
from InferAbs2Titile import InferAbs2Title
model = InferAbs2Title('m-aliabbas/model-t51-base1')
abstract="""

"""
title = model(abstract)
print(title)
```
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5.6e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Framework versions

- Transformers 4.27.2
- Pytorch 1.12.1+cu116
- Datasets 2.4.0
- Tokenizers 0.12.1
