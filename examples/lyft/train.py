
import matplotlib as mpl

mpl.use("Agg")
from nncore.core.opt import Opts
from nncore.segmentation.pipeline import Pipeline

if __name__ == "__main__":
    opt = Opts(cfg_path="./opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()
