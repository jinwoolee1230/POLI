import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_preprocess.pcd_data_class import PCD_Dataset
from model.pointnetpp_scene import PointPP
from sdprlayers import SDPPoseEstimator

from utils import covariance_generation, sigmoid_mask, to_homogeneous_xyz, make_loss, scale_aware_normalization

def parse_args():
    parser = argparse.ArgumentParser(description='Sample Dataset Training Script')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--logdir', type=str, required=True, help='Path to log directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--mean', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Normalization mean for scale-aware normalization')
    parser.add_argument('--scaling_factor', type = float, default = 100.0, help='Scaling factor for scale-aware normalization')
    return parser.parse_args()

def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # PARAMETERS
    weight_reg = 1000000.0
    logdet_reg_weight = 0.01/weight_reg
    corr_loss_weight = 1.0/weight_reg
    pose_loss_weight = 1.0

    # region #* dataset loading, model setup

    dataset_dir = args.dataset_dir #* TODO: set your dataset directory
    dataset = PCD_Dataset(dataset_dir)

    train_dataloader = DataLoader(dataset, batch_size= 1, shuffle=True)
    
    model = PointPP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    T_s_v = torch.eye(4, device='cuda', dtype=torch.float32)
    sdpr_block = SDPPoseEstimator(T_s_v).to('cuda')
    # endregion

    # region #* initialization and checkpoint loading
    step = 0
    valid_step = 0
    epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        valid_step = checkpoint['valid_step']
    
    writer = SummaryWriter(log_dir=args.logdir)

    # endregion
    
    for iter in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, desc="Train Progress"):
            loss = 0.0
            #region #* data loading and preprocessing

            P = batch['P'].to('cuda')               # [B, 3, N]
            Q = batch['Q'].to('cuda')               # [B, 3, M]
            R_rel = batch['R_rel'].to('cuda')       # [B, 3, 3]
            t_rel = batch['t_rel'].to('cuda')       # [B, 3, 1]
            indices = batch['indices'].to('cuda')   # [B, N]
            dists = batch['dists'].to('cuda')       # [B, N]
            
            B = P.shape[0]
            N = P.shape[2]
            normalized_P, normalized_Q = scale_aware_normalization(P, Q, args.mean, args.scaling_factor) # [B, 3, N], [B, 3, M]

            # endregion

            # region #* model inference and covariance generation
            C_p = model(normalized_P)                          # [B, 6, N]
            C_q = model(normalized_Q)                          # [B, 6, M]
            C_p= C_p.permute(0, 2, 1)                          # [B, N, 6]
            C_q= C_q.permute(0, 2, 1)                          # [B, M, 6]
            C_p= covariance_generation(C_p)                    # [B, N, 3, 3]
            C_q= covariance_generation(C_q)                    # [B, M, 3, 3]
            # endregion

            # region #* correspondence matching
            P = P.permute(0, 2, 1)                        # [B, N, 3]
            Q = Q.permute(0, 2, 1)                        # [B, M, 3]
            batch_idx = torch.arange(B, device=P.device).unsqueeze(-1)
            C_q_matched = C_q[batch_idx, indices]   # [B, N, 3, 3]
            Q_matched   = Q[batch_idx, indices]     # [B, N, 3]
            # endregion
            
            # region #* information matrix generation
            information_matrix = torch.inverse(2*C_q_matched) # [B, N, 3, 3]
            information_matrix = information_matrix.to(P.device)
            # endregion

            # region #* SDPRLayer pose estimation
            mask = sigmoid_mask(dists, threshold=2.0, alpha=30) # [B, N]
            P_masked = P * mask.unsqueeze(-1)                     # [B, N, 3]
            Q_matched_masked = Q_matched * mask.unsqueeze(-1)     # [B, N, 3]

            information_matrix_masked = information_matrix * mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 3, 3]
            P_hm = to_homogeneous_xyz(P_masked)                    # [B, 4, N]
            Q_hm = to_homogeneous_xyz(Q_matched_masked)            # [B, 4, N]
            weights = torch.ones(B, 1, N, device=P.device)         # [B, 1, N]

            try:
                T_est, ER_loss = sdpr_block(
                    keypoints_3D_src=P_hm,
                    keypoints_3D_trg=Q_hm,
                    weights=weights,
                    inv_cov_weights=information_matrix_masked
                ) # [B, 4, 4]
                if T_est is None:
                    print("Warning: T_est is None.")
                    continue
            except Exception as e:
                print("Warning: SDPR failed.")
                continue
            # endregion
            
            # region #* pose likelihood 

            R_rel_est = T_est[:, :3, :3]                               # [B, 3, 3]
            t_rel_est = T_est[:, :3, 3:]                               # [B, 3, 1]
            pose_loss = pose_loss_weight * make_loss(R_rel_est, t_rel_est, R_rel, t_rel) # [B]
            loss += pose_loss

            # endregion
            
            # region #* correspondence likelihood

            #* correspondence loss
            Q_hat = R_rel_est @ P.permute(0,2,1) + t_rel_est     # [B, 3, N]
            error = Q_matched.permute(0,2,1) - Q_hat             # [B, 3, N]
            error_vec = error.permute(0,2,1).unsqueeze(-1)       # [B, N, 3, 1]
            Ie = information_matrix @ error_vec                  # [B, N, 3, 1]
            eTIe = error_vec.permute(0,1,3,2) @ Ie               # [B, N, 1, 1]
            weighted_eTIe = eTIe.squeeze(-1).squeeze(-1) * mask  # [B, N]
            corr_loss = corr_loss_weight * weighted_eTIe.sum()   # [B]
            loss += corr_loss
        
            #* logdet regularization
            reg = torch.logdet(information_matrix)               # [B, N]
            masked_reg = -1* logdet_reg_weight * reg * mask      # [B, N]
            masked_reg = masked_reg.sum()
            loss += masked_reg

            #endregion
                
            writer.add_scalar('Train_step/pose_loss', pose_loss.item(), step)
            writer.add_scalar('Train_step/logdet_reg_loss', masked_reg.mean().item(), step)
            writer.add_scalar('Train_step/corr_loss', corr_loss.item(), step)
            writer.add_scalar('Train_step/total_loss', loss.item(), step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()

        writer.add_scalar('Train_epoch/total_loss', total_loss / len(train_dataloader), epoch)
        epoch += 1
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'valid_step': valid_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.logdir, f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
