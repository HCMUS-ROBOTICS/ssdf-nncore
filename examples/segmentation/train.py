import sys
from importlib import import_module

sys.path.insert(0, "../../")

from nncore.opt import opts
from nncore.pipeline import Pipeline

if __name__ == "__main__":
    opt = opts(cfg_path="./config/opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()
