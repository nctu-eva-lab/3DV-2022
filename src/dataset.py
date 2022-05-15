import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

import os
import glob
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesVertex

class ShapeNetDB(Dataset):
    def __init__(self, data_dir, data_type, img_transform=False):
        super(ShapeNetDB).__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.db = self.load_db()
        self.img_transform = img_transform

        self.get_index()


    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        if self.data_type == 'point':
            """
            Return shapes:
            img: (B, 256, 256, 3)
            pc: (B, 2048, 3)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            pc, object_id = self.load_point(idx)

            assert img_id == object_id

            return img, pc, object_id
        
        elif self.data_type == 'voxel':
            """
            Return shapes:
            img: (B, 256, 256, 3)
            voxel: (B, 33, 33, 33)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            voxel, object_id = self.load_voxel(idx)

            assert img_id == object_id

            return img, voxel, object_id

        # elif self.data_type == 'mesh':
        #     img, img_id = self.load_img(idx)
        #     mesh, object_id = self.load_mesh(idx)

        #     assert img_id == object_id

        #     return img, mesh, object_id

    def load_db(self):
        # print(os.path.join(self.data_dir, '*'))
        db_list = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        # print(db_list)

        return db_list
    
    def get_index(self):
        self.id_index = self.data_dir.split('/').index("data") + 2
        # print(self.id_index)

    def load_img(self, idx):
        path = os.path.join(self.db[idx], 'view.png')
        img = read_image(path) / 255.0
        img = img.permute(1,2,0)
        # raw_img = Image.open(path)
        # img = torch.from_numpy(np.array(raw_img) / 255.0)[..., :3]
        # img = img.to(dtype=torch.float32)

        # if self.img_transform:
        #     trans = transforms.Compose([
        #                                 transforms.Resize(512),
        #                                 transforms.ToTensor()
        #                                 ])
        #     img = trans(img)

        object_id = self.db[idx].split('/')[self.id_index]

        return img, object_id
    
    # def load_mesh(self, idx):
    #     path = os.path.join(self.db[idx], 'model.obj')
    #     verts, faces, _ = load_obj(path, load_textures=False)
    #     faces_idx = faces.verts_idx

    #     # normalize
    #     center = verts.mean(0)
    #     verts = verts - center
    #     scale = max(verts.abs().max(0)[0])
    #     verts = verts / scale

        # make white texturre
        # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        # textures = TexturesVertex(verts_features=verts_rgb)

        # mesh = Meshes(
        #     verts=[verts],
        #     faces=[faces_idx],
        #     textures=textures
        # )

        # object_id = self.db[idx].split('/')[self.id_index]

        # return mesh, object_id

    def load_point(self, idx):
        path = os.path.join(self.db[idx], 'point_cloud.npy')
        points = np.load(path)

        # resample
        # n_points = 2048
        # choice = np.random.choice(points.shape[0], n_points, replace=True)
        # points = points[choice, :3]

        # normalize
        points = points - np.expand_dims(np.mean(points, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis = 1)),0)
        points = points / dist #scale

        object_id = self.db[idx].split('/')[self.id_index]

        return torch.from_numpy(points), object_id
    
    def load_voxel(self, idx):
        path = os.path.join(self.db[idx], 'voxel.npy')
        voxel = np.load(path)

        object_id = self.db[idx].split('/')[self.id_index]

        return torch.from_numpy(voxel), object_id


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # from pytorch3d.datasets import collate_batched_meshes

    db = ShapeNetDB('/home/odie/3dv-hw/data/chair', 'point')
    dataloader = DataLoader(db, batch_size=10, shuffle=True)

    for img, point, object_id in dataloader:
        print(img.shape)
        print(point.shape)   
        print(object_id)
        break