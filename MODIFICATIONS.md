# Modifications from Original dLLM

## Files Changed

### Core Changes
1. **dllm/core/trainers/mdlm.py**
   - import numpy and math (16 - 19)
   - Adding `frequency_dict`to estimate the global surprise thanks to the dataset (115)
   - Adding Temperature hyperparameter (116)
   - Adding the new logic of masking instead of making it uniform (137 - 175)
   

## New Files Added
- `dllm/train/train_egdllm.py`: Training pipeline