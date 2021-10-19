from dataset import Cam2BEVDataset  # noqa
from learner import Cam2BEVLearner  # noqa
from model import build_deeplabv3_cam2bev  # noqa

from nncore.core.opt import Opts
from nncore.segmentation.pipeline import Pipeline

if __name__ == "__main__":
    opt = Opts(cfg_path="opt.yaml").parse()
    train_pipeline = Pipeline(opt)
    train_pipeline.fit()
