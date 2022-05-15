from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
# from pytorch3d.utils import ico_sphere
# import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, cfg):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[cfg.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if cfg.dtype == "voxel":
            pass
            # TODO:
            # self.decoder =             
        elif cfg.dtype == "point":
            self.n_point = cfg.n_points
            # TODO:
            # self.decoder = PointDecoder(cfg.n_points, 512)
        # elif cfg.dtype == "mesh":
        #     # try different mesh initializations
        #     mesh_pred = ico_sphere(4,'cuda')
        #     self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*cfg.batch_size, mesh_pred.faces_list()*cfg.batch_size)
        #     # TODO:
        #     # self.decoder =             

    def forward(self, images, cfg):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        # call decoder
        if cfg.dtype == "voxel":
            # TODO:
            # voxels_pred =             
            return voxels_pred

        elif cfg.dtype == "point":
            # TODO:
            # pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        # elif cfg.dtype == "mesh":
        #     # TODO:
        #     # deform_vertices_pred =             
        #     mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
        #     return  mesh_pred          


# class PointDecoder(nn.Module):
#     def __init__(self, num_points, latent_size):
#         super(PointDecoder, self).__init__()
#         self.num_points = num_points
#         self.fc0 = nn.Linear(latent_size, 100)
#         self.fc1 = nn.Linear(100, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.fc5 = nn.Linear(1024, self.num_points * 3)
#         self.th = nn.Tanh()

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.th(self.fc5(x))
#         x = x.view(batchsize, self.num_points, 3)
#         return x