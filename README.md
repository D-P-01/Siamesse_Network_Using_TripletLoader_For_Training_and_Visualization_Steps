# Siamesse_Network_Using_TripletLoader_For_Training_and_Visualization_For_Custom_Dataset_Steps


## Steps for Training on a Custom Dataset:

Follow these steps to train the model on your custom dataset:

```bash
# 1. Clone the GitHub Repository
git clone https://github.com/avilash/pytorch-siamese-triplet

# 2. Navigate to the Cloned Directory
cd Path_to_your_directory

# 3. Install the Required Dependencies
pip install -r requirements.txt

# 4. Modify test.yaml for Custom Dataset
# Open the test.yaml file in the config folder and add the path to your custom dataset:
# CUSTOM:
#   HOME: "Path to the custom dataset"

# 5. Ensure the directory structure is as follows:
# +-- root
# |   +-- train
# |       +-- class1
# |           +-- img1.jpg
# |           +-- img2.jpg
# |           +-- img3.jpg
# |       +-- class2
# |       +-- class3
# |   +-- test
# |       +-- class4
# |       +-- class5

# 6. Set Custom Dataset in train.py
# In the train.py file, set the default dataset as custom to use your dataset for training.

# 7. Run the Training Script
python train.py --result_dir results --exp_name Custom_exp1 --cuda --epochs 10 --ckp_freq 5 --dataset custom --num_train_samples 5975 --num_test_samples 500 --train_log_step 50

# 8. Check Results
# The results will be stored as a checkpoint.pth file and model weights will be saved after every 5 iterations by default.
```

## Steps for t-SNE Visualization of Custom Dataset:

Follow these steps for doing tsne visulisation on your custom dataset:

```bash
# 1. Replace the existing `tsne.py` file with your modified `tsne.py` file
# The modified `tsne.py` is capable of handling custom dataset conditions.

# 2. Run the following command to visualize t-SNE results:
python tsne.py --ckp 'Path to your trained model.pth file'

# 3. The t-SNE visualization image and .pkl file will be saved in the `data` folder by default.
```

## Steps for generating Confusion Matrix:

Follow these steps:

```bash
# 1. Replace custom_dset.py file load fun with given code:
#  def load(self,dir):
        base_path = dir
        super(Custom, self).load(base_path)

# 2. Add the confusion_matrix.py file in the cloned repo and run the file :
