<div align="center">

**Collection of tasks for fast baseline solving in self-driving problems with deep learning. The offical language support is Pytorch**

---

<p align="center">
  <a href="#what-is-nncore">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#license">License</a>
</p>

</div>

---

**Note:** This pkg is currently in development. Please [open an issue](https://github.com/HCMUS-ROBOTICS/ssdf-nncore/issues/new/choose) if you find anything that isn't working as expected.

---

## What is NNcore

NNcore is a deep learning framework focusing on solving autonomous-driving problems.

- Task-based training and export for multiple framework

## Installation

Pip / conda

```bash
pip install --upgrade --force-reinstall --no-deps albumentations
pip install qudida
pip install git+https://github.com/HCMUS-ROBOTICS/ssdf-nncore
```

<details>
<summary>Other installations</summary>

To install **nncore** and develop locally

```bash
git clone https://github.com/HCMUS-ROBOTICS/ssdf-nncore nncore
cd nncore
pip install -e .
```
</details>

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

## Perform inference

### Export PyTorch checkpoint to ONNX

Use the script provided in `examples`, then run the below command

```bash
python3 torch2onnx.py <checkpoint> --in_shape 1 3 224 224 --inputs input --outputs output
```

### Performing inference

See [serve](serve) library

## Contribution

If you want to contribute to `nncore`, please follow steps below:
1. Fork your own version from this repository
1. Checkout to another branch, e.g. `fix-loss`, `add-feat`.
1. Make changes/Add features/Fix bugs
1. Add test cases in the `test` folder and run them to make sure they are all passed (see below)
1. Run code format to check formating before making a commit (see below)
1. Push the commit(s) to your own repository
1. Create a pull request

**To run tests**
```bash
pip install pytest
python -m pytest test/
```

**To run code-format**
```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```

## License
See [LICENSE](LICENSE)
