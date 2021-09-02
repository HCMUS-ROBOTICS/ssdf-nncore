import argparse

import torch
import tqdm
from models import *
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import *
import torch.nn.functional as F

random_seed = 213
batch_size = 64
lr = 0.001
n_epochs = 5

transform_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


model = MobileUnet().to("cuda")
criterion = CrossEntropyLoss().to("cuda")
optimizer = Adam(model.parameters(), lr=lr)

train_data = dira20("./data/", train=True)
# val_data = dira20('/home/ken/Documents/test_tensorRT/dataset/', train=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=batch_size)

running_loss = 0.0
for e in tqdm.tqdm(range(n_epochs)):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
    print(loss)
    # e.update()
print("Finished Training {}".format(loss))

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="baseline", help="Name of file")
parser.add_argument(
    "--onnx",
    action="store_true",
    default=False,
    help="Save in ONNX format, i.e. baseline.onnx",
)
args = parser.parse_args()
print(args)

torch.save(model.state_dict(), f"{args.name}.pth")
if args.onnx:
    model.eval()  # important
    input_names = ["input"]
    output_names = ["output"]
    with torch.no_grad():
        dummy_input = torch.autograd.Variable(torch.rand(1, 3, 224, 224).cuda())
        torch.onnx.export(
            model,
            dummy_input,
            f"{args.name}.onnx",
            verbose=True,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
        )

print("Done.")
# model.eval()
# im = model(train_data[0][0].to('cuda').unsqueeze(0))[0].cpu()
# print(im.shape)
# transforms.ToPILImage(mode='L')(im).save('train.jpg')
