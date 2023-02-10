# CISC881
Lessons for CISC 881 - Deep Learning for Medical Imaging

## Quickstart
First, clone this repository: 
```bash 
git clone https://github.com/med-i-lab/CISC881
chdir CISC881
```
create an environment for the project (we recommend anaconda), and install the requirements: 
```bash
conda create --name cisc881 python=3.9
conda activate cisc881
pip install -r requirements.txt
```

## Pytorch Basics
We use pytorch which is the most widely used framework for deep learning researchers. If you are not familiar with pytorch, that's fine! Please check out this video https://www.youtube.com/watch?v=IC0_FRiX-sw&ab_channel=PyTorch for an introduction to pytorch. We also included a "Hello world" pytorch example, which can be run using:
```bash
python basic_training_example.py
```
Try running the example and reading through the script to get a feel for it.

## Chest X-ray Lesson

We will be using a chest X-ray dataset as the source for our lessons. We need to download the dataset, it is available at https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia. To follow along with the demo, you need to download the dataset, put it in the `data/` folder, and unzip it. 
