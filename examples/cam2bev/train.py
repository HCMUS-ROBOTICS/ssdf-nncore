from dataset import Cam2BEVDataset  # noqa
from learner import Cam2BEVLearner  # noqa
from model import build_deeplabv3_cam2bev  # noqa

from nncore.core.opt import Opts
from nncore.segmentation.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts.parse('opt.yaml')
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()
