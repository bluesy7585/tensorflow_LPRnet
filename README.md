# tensorflow LPRnet
tensorflow implementation of LPRnet. A lightweight deep network for number plate recognition.

- multiple scale CNN features
- CTC for variable length chars
- no RNN layers

## training
generate plate images for training

`python gen_plates.py`

generate validation images

`python gen_plates.py -s .\valid -n 200`

train

`python main.py -m train`

or train with runtime-generated images

`python main.py -m train -r`

model checkpoint will be save for each `SAVE_STEPS` steps.
validation will be perform for each `VALIDATE_EPOCHS` epochs.

## test
generate test images

`python gen_plates.py -s .\test -n 200`

restore checkpoint for test

`python main.py -m test -c [checkpioint]`

e.g

```
python main.py -m test -c .\checkpoint\python main.py -m test -c .\checkpoint\LPRnet_steps8000_loss_0.069.ckpt
...
val loss: 0.31266
plate accuracy: 192-200 0.960, char accuracy: 1105-1115 0.99103
```

### test single image

to test single image and show result

`python main.py -m test -c [checkpoint] --img [image fullpath]`

e.g
```
python main.py -m test -c .\checkpoint\LPRnet_steps5000_loss_0.215.ckpt --img .\test\AW73RHW_18771.jpg
...
restore from checkpoint: .\checkpoint\LPRnet_steps5000_loss_0.215.ckpt
AM73RHW
```
## train custom data
change `TRAIN_DIR`, `VAL_DIR` in LPRnet.py to training/validation data folder

image filename with the format [label]_XXXX

e.g AB12CD_0000.jpg

- char set

  change `CHARS` if possible chars in label is different with default.

- char length

  default input resolution (94x24) has 24 timesteps in CTC layer.

  if your data have more than 8 chars in images, perhaps use wider resolution for good performance. 

  e.g input width 128 has 32 timesteps in CTC layer.

## references
- [LPRnet](https://arxiv.org/abs/1806.10447 "LPRnet")
- https://github.com/lyl8213/Plate_Recognition-LPRnet
- https://github.com/sirius-ai/LPRNet_Pytorch
- https://github.com/mahavird/my_deep_anpr
