

from nncore.opt import opts
from nncore.pipeline import Pipeline

if __name__ == "__main__":
    opt = opts(cfg_path="./opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()
