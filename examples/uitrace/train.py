import matplotlib as mpl

mpl.use("Agg")
from nncore.core.opt import Opts
from nncore.segmentation.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts.parse("opt.yaml")
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()
