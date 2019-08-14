# tensorflow LPRnet
tensorflow implementation of LPRnet. A lightweight deep network for number plate recognition.

## training
generate plate images for training

`python gen_plates.py`

generate validation images

`python gen_plates.py -s .\valid -n 200`

train

`python main.py -m train`

or train with runtime-generated images

`python main.py -m train -r`
## test
generate test images

`python gen_plates.py -s .\test -n 200`

restore checkpoint for test

`python main.py -m test -c [checkpioint]`

e.g

`python main.py -m test -c .\checkpoint\LPRnet_steps5000_loss_0.215.ckpt`

## references
- [LPRnet](https://arxiv.org/abs/1806.10447 "LPRnet")
- https://github.com/lyl8213/Plate_Recognition-LPRnet
- https://github.com/sirius-ai/LPRNet_Pytorch
- https://github.com/mahavird/my_deep_anpr
