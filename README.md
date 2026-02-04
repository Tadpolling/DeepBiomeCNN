# DeepBiomeCNN
This project is a CNN model whose goal it is to classify minecraft biomes. 

## Setup
Start by cloning this repo and installing the dependencies
```bash
git clone https://github.com/Tadpolling/DeepBiomeCNN.git
cd DeepBiomeCNN
pip install -r requirements.txt
```

link to download train set (Google Drive), download the trainset and extract in the data folder.

[Download here](https://drive.google.com/drive/folders/1Sn-OYBVNYAagYASFGuup7g6W_sNy85Kn?usp=sharing) 

Now in order to train the model run the follow script:
```bash
python src/train.py
```
In order to test the model run 
```bash
python src/test.py
```
This runs the model that was trained previously. Do note that the training scripts try and find a model if it exists so you need to delete the ".pth" file in order to create a completely new model. 
The train and test script create images in the src folder. 
