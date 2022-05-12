import os
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftGouraudShader,
    TexturesUV,
    TexturesVertex
)

# from .utils import load_mesh

def load_mesh(obj, device):
    verts, faces_idx, _ = load_obj(obj, load_textures=False)
    faces = faces_idx.verts_idx

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    return mesh


class Scene():
    def __init__(self, device):
        self.device = device
        
    def set_cam(self, dist=1.0, elev=0.0, azim=0.0):
        """
        Initialize a camera.
        With world coordinates +Y up, +X left and +Z in.
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_view_transform
        """
        R, T = look_at_view_transform(dist, elev, azim) 
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)   
    
    def set_light(self, location=[[0.0, 0.0, 0.0]]):
        """
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/lighting.html#pytorch3d.renderer.lighting.PointLights
        """
        self.lights = PointLights(location=location, device=self.device)

    def set_rasterizer(self, image_size=512):
        self.raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=1e-5, 
            faces_per_pixel=150,
            bin_size=0,
            cull_backfaces=True,
            # max_faces_per_bin=10,
        )
        """
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/rasterizer.html#pytorch3d.renderer.mesh.rasterizer.RasterizationSettings
        """
    
    def set_renderer(self):
        """
        see full api in https://github.com/facebookresearch/pytorch3d/blob/2c64635daa2aa728f35ed4abe41c6942ae8c0d8b/pytorch3d/renderer/mesh/renderer.py#L32
        """

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftGouraudShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )

        return self.renderer


if __name__ == '__main__':

    mesh = load_mesh(obj="/home/odie/3dv-hw/notebooks/d9061907b7b411dbe3db80a3cacc6e3/model.obj", 
                    device="cuda:0")

    scene = Scene("cuda:0")
    scene.set_cam(0.8, 0, 90)
    scene.set_light(location=[[0.0, 0.0, 0.0]])
    scene.set_rasterizer(image_size=256)
    scene.set_renderer()

    images = scene.renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.savefig('foo.png')
