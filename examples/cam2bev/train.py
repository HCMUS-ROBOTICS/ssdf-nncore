from nncore.core.opt import opts
from nncore.segmentation.pipeline import Pipeline
from learner import Cam2BEVLearner  # noqa
from dataset import Cam2BEVDataset  # noqa
from model import build_deeplabv3_cam2bev  # noqa


if __name__ == "__main__":
    opt = opts(cfg_path="opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()
