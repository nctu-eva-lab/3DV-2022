import time
import torch
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
import src.losses as losses

import hydra
from omegaconf import DictConfig


def calculate_loss(predictions, ground_truth, cfg):
    if cfg.type == 'voxel':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif cfg.type == 'point':
        loss = losses.chamfer_loss(predictions, ground_truth)
    # elif cfg.type == 'mesh':
    #     sample_trg = sample_points_from_meshes(ground_truth, cfg.n_points)
    #     sample_pred = sample_points_from_meshes(predictions, cfg.n_points)

    #     loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
    #     loss_smooth = losses.smoothness_loss(predictions)

        # loss = cfg.w_chamfer * loss_reg + cfg.w_smooth * loss_smooth        
    return loss

@hydra.main(config_path="configs/", config_name="config")
def evaluate_model(cfg: DictConfig):
    shapenetdb = ShapeNetDB('./data/chair', 'point')

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_loss = []

    if cfg.load_eval_checkpoint:
        checkpoint = torch.load(f'checkpoint_{cfg.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    for step in range(start_iter, cfg.max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        images_gt, ground_truth_3d, _ = next(eval_loader)
        images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, cfg)

        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()

        # TODO:
        # if (step % cfg.vis_freq) == 0:
        #     # visualization block
        #     #  rend = 
        #     plt.imsave(f'vis/{step}_{args.type}.png', rend)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        avg_loss.append(loss)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); eva_loss: %.3f" % (step, cfg.max_iter, total_time, read_time, iter_time, torch.tensor(avg_loss).mean()))

    print('Done!')

if __name__ == '__main__':
    evaluate_model()