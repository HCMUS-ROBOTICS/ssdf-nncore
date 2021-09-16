import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image


class TensorboardLogger:
    def __init__(self, path):
        assert path != None, "path is None"
        self.writer = SummaryWriter(log_dir=path)

    def update_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def update_loss(self, phase, value, step):
        self.update_scalar(f"{phase}/loss", value, step)

    def update_metric(self, phase, metric, value, step):
        self.update_scalar(f"{phase}/{metric}", value, step)

    def update_lr(self, gid, value, step):
        self.update_scalar(f"lr/group_{gid}", value, step)

    def update_figure(self, tag, image, step):
        self.writer.add_figure(tag, image, global_step=step)


def image_batch_show(batch, ncol=5, fig_size=(30, 10)):
    # https://stackoverflow.com/questions/51329159/how-can-i-generate-and-display-a-grid-of-images-in-pytorch-with-plt-imshow-and-t
    grid_img = torchvision.utils.make_grid(batch, nrow=ncol)
    # grid_img = grid_img.permute(1, 2, 0)
    return grid_img.float()
    # fig = plt.figure(figsize=fig_size)
    # if show:

    # plt.plot(grid_img)
    # else:
    #     _ = plt.plot(grid_img)
    # return plt.gcf()


def image_batch_show_with_title(batch, title=[], ncol=5, fig_size=(30, 10), show=False):
    NotImplemented
