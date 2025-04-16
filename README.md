# mri_finetune
Machine learning to classify graded ligament tears


## Data and Dependencies

To download the MRNet dataset and install dependencies use the `data_dependencies.ipynb` file.

## Model Training

The model was trained in the `train.ipynb` file using a framework from https://github.com/spmallick/learnopencv/blob/master/MRnet-MultiTask-Approach/train.py.

## Model Tuning

The model can be further tuned for other MRI classification tasks using the `weights/combined` folder. The files are named for their metric performance values.