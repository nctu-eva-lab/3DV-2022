from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
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
        if cfg.type == "voxel":
            pass
            # TODO:
            # self.decoder =             
        elif cfg.type == "point":
            self.n_point = cfg.n_points
            # TODO:
            # self.decoder =             
        # elif cfg.type == "mesh":
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
        if cfg.type == "voxel":
            # TODO:
            # voxels_pred =             
            return voxels_pred

        elif cfg.type == "point":
            # TODO:
            # pointclouds_pred =             
            return pointclouds_pred

        # elif cfg.type == "mesh":
        #     # TODO:
        #     # deform_vertices_pred =             
        #     mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
        #     return  mesh_pred          