# ssdf-nncore 

This is neural net framework for SSDF tasks. The offical language support is Pytorch

# Installation 

To install **nncore** and develop locally

```
git clone https://github.com/HCMUS-ROBOTICS/ssdf-nncore nncore 
cd nncore/nn
pip install -r requirements.txt
```
# Segmentation task 
## Workspace structure 
Setup data have the same structure below.

```
this repo
│
└─── nn
│    └─── data
│       └─── images
│            │     00000.jpg
│            │     00001.jpg
│            │     ...
│       └─── mask
│            │     00000.jpg
│            │     00001.jpg
│            │     ...
|    train.py
|    test.py
```

## Evaluation 

For evaluate score, checkout example testing scripts `test.py` 

Optional arguments:

- `--model-path `: path to pretrained model
- `--data-path`: path to data root folder
- `--img-folder-name`: image folder name           
- `--msk-folder-name`: mask / label folder name
- `--train`: training flag, not use in inference mode
- `--extension`: image extenstion (example: png, jpg)
                        
Example: 
```
python test.py \
--data ./data \
--img images \
--msk mask \
--model ./tmp/best_loss.pth \
--train
```

## Train your own model


For training new model, checkout example training scripts `train.py`. Most of arguments are similar with `test.py`

Optional arguments:

- `--model`: model class (only support MobileUnet now)
- `--data-path`: path to data root folder
- `--img-folder-name`: image folder name           
- `--msk-folder-name`: mask / label folder name
- `--train`: training flag, not use in inference mode
- `--extension`: image extenstion (example: png, jpg)


Example: 
```
python train.py \
--model MobileUnet \
--data ./data \
--img images \
--msk mask \
--train
```