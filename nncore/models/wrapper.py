import torch


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch["input"])
        loss, loss_dict = self.loss(outputs, batch)
        return outputs, loss, loss_dict

