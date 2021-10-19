<div align="center">

**Collection of tasks for fast baseline solving in self-driving problems with deep learning. The offical language support is Pytorch**

---

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#what-is-nncore">About</a> •
  <a href="#inference">Prediction</a> •
  <a href="#training">Training</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#a-general-task">General Task</a> •
  <a href="#license">License</a>
</p>

</div>

---

**Note:** This pkg is currently in development. Please [open an issue](https://github.com/HCMUS-ROBOTICS/ssdf-nncore/issues/new/choose) if you find anything that isn't working as expected.

---

## Installation

Pip / conda

```bash
non-support yet
```

<details>
  <summary>Other installations</summary>

To install **nncore** and develop locally

```bash
git clone https://github.com/HCMUS-ROBOTICS/ssdf-nncore nncore
cd nncore/nn
pip install -r requirements.txt
```

See [Installation]() for more options.

</details>

---

## What is NNcore

NNcore is a deep learning framework focusing on solving autonomous-driving problems.

- Task-based training and export for multiple framework

### Predictions

this part not work yet

```python
from nncore.utils.getter import get_instance
from nncore.utils import load_model, load_yaml, get_device, image_batch_show
from nncore.models.wrapper import SegmentationModel
from nncore.utils.segmentation import multi_class_prediction
from torchvision.utils import save_image


from pathlib import Path

if __name__ == "__main__":
    checkpoint_folder = Path("./path/to/checkpointfolder")
    cfg = load_yaml(checkpoint_folder / "config.yaml")
    model = get_instance(cfg["pipeline"]["model"])
    load_model(model, checkpoint_folder / "best_loss.pth")
    device = get_device()
    inference_model = SegmentationModel(model, None)
    rgbs, pred = inference_model.predict(
        [
            "./data/images/00001.png",
            "./data/images/00002.png",
            "./data/images/00003.png",
            "./data/images/00004.png",
        ],
        device=device,
        batch_size=2,
        return_inp=True,
    )

```

### Training

<details>
  <summary>training configs</summary>
cfg/opt.yaml

```yaml
opts:
  pretrained: null # untested yet
  id: default

  debug: True # if debug = true, model will not save checkpoint, untested yet
  demo: False # not support yet
  resume: False # not support yet
  test: False # untested yet

  nepochs: # number of epoch

  gpus: 0,1,2,3 # not support yet
  num_workers: # worker num
  fp16: True # untested yet

  val_step: # validate freq
  log_step: # log freq

  num_iters: -1 # unsupport yet
  save_dir: # save directory (sample images, checkpoints, cfg)
  verbose: # if verbose is False, no console logging during training
  seed: # fixed random seed
  cfg_pipeline: # path to pipeline.yaml

```

pipeline.yaml

```yaml
learner:
  name: # learner name
  args:
device:
  name: # not support yet
  args:
model:
  name: # model name
  args:
criterion:
  name: # loss name
  args:
optimizer:
  name: # optimizer name
  args:
metric:
  - name: # metric names
    args:
scheduler:
  name: # scheduler lr name
  args:
data:
  # optional, if train and val is not Null, pipeline will use your dataset directly
  train: # dataset name
    name:
    args:
    loader:
      name: DataLoader
      args:
        batch_size: 16
        shuffle: True
        drop_last: True
  val:
    name: # dataset name
    args:
    loader:
      name: DataLoader
      args:
        batch_size: 16
        shuffle: True
        drop_last: False
  # optional, if trainval is not Null, train and val is Null, pipeline will split your dataset
  trainval:
    test_ratio: 0.2
    dataset:
      name: #dataset name
      constructor: # constructor name if default is not init
      args:
    loader:
      train:
        name: # train data loader
        args:
      val:
        name: # val data loader
        args:
```

</details>

train.py

```python

from nncore.opt import Opts
from nncore.pipeline import Pipeline

if __name__ == "__main__":
    opt = Opts(cfg_path="'cfg/opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()

```

Then use the evaluate model:

```python

train_pipeline.evaluate()

```

---

## Tasks

### Example 1: Semantic Segmentation

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

more information [here](examples/segmentation)

## Export to ONNX

Use the script provided in `examples`, then run below command

```bash
PYTHONPATH=.. python3 torch2onnx.py <checkpoint> --in_shape 1 3 224 224 --inputs input --outputs output
```

## A general task

Nothing here yet

## Customizable

Nothing here yet

## Visualization

Predictions from vision tasks can be visualized through an [Tensorboard](https://www.tensorflow.org/tensorboard), allowing you to better understand and analyze how your model is performing.

## References

Nothing here yet

## License
