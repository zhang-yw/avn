import torch
import torchvision
import torchvision.transforms as transforms
import h5py
import os


# prepare model


def hook(module, input, output):
    fc7_features.append(output.clone().detach())


model = torchvision.models.resnet50(pretrained=True)
handle = model.avgpool.register_forward_hook(hook)


# prepare data
for scene_name in os.listdir("./data/environment/"):
    if not scene_name.endswith(".h5"):
        continue
    fc7_features = []
    # scene_name = "nnew1.h5"
    print(scene_name)
    in_path = "./data/environment/%s" % scene_name
    h5_file_in = h5py.File(in_path, 'r+')
    obs = h5_file_in["observation"]
    print(obs.shape)

    for idx in range(len(obs)):
        image = obs[idx]
        image = transforms.functional.to_tensor(image)
        image = image.unsqueeze(0)
        out = model(image)

    fc7_features = torch.cat(fc7_features, dim=0).squeeze()
    fc7_features = fc7_features.numpy()

    if "resnet_feature" in h5_file_in.keys():
        h5_file_in.__delitem__("resnet_feature")
    h5_file_in.create_dataset("resnet_feature", data=fc7_features)
