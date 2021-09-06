from nncore.opt import opts
from nncore.pipeline import Pipeline

if __name__ == "__main__":
    opt = opts().parse()
    train_pipeline = Pipeline(opt, cfg_path="./configs/default/pipeline.yaml")
    train_pipeline.fit()
