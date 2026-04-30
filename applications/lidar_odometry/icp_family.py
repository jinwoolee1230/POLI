import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
import torch
import math
from data_preprocess.pcd_data_class import PCD_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch_kdtree import build_kd_tree as _build_kd_tree
    _TORCH_KDTREE_IMPORT_ERROR = None
except Exception as exc:
    _build_kd_tree = None
    _TORCH_KDTREE_IMPORT_ERROR = exc


def build_kd_tree(*args, **kwargs):
    if _build_kd_tree is None:
        raise ImportError(
            "`torch_kdtree` is required for this ICP routine, but it could not be imported. "
            "Use a routine that does not rely on KD-tree search, or fix the torch_kdtree installation."
        ) from _TORCH_KDTREE_IMPORT_ERROR
    return _build_kd_tree(*args, **kwargs)


# ✅ Naive point-to-point ICP
def po2po(test_folder : str) -> list:
    dataset = PCD_Dataset(test_folder)     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    
    R_prev = torch.eye(3).to('cuda')
    t_prev = torch.zeros((3, 1)).to('cuda')
    
    for batch in tqdm(dataloader, desc="Test po2po Progress"):
        P = batch['P'].to('cuda')                   # (1, 3, 10000)
        Q = batch['Q'].to('cuda')                   # (1, 3, 10000)
        R_rel_gt = batch['R_rel'].to('cuda')        # (1, 3, 3)
        t_rel_gt = batch['t_rel'].to('cuda')        # (1, 3, 1)

        P = P.squeeze(0)                             # (3, 10000)
        Q = Q.squeeze(0)                             # (3, 10000)

        R_est, t_est = icp_point2point_gauss_newton(P, Q, initial_R = R_prev, initial_t = t_prev)
        R_prev = R_est.clone() # \hat{Q_R_P}, (3, 3)
        t_prev = t_est.clone() # \hat{Q_t_P}, (3, 1)

        R_gt = R_rel_gt.squeeze() # (3, 3)
        t_gt = t_rel_gt.squeeze(0) # (3, 1)


        results.append([[R_est.cpu(), t_est.cpu()], [R_gt.cpu(), t_gt.cpu()]])

    return results

# ✅ Gaussian-Newton point-to-point ICP
def icp_point2point_gauss_newton(P : torch.tensor,
                                 Q : torch.tensor,
                                 initial_R : torch.tensor,
                                 initial_t : torch.tensor,
                                 max_iterations : int = 50,
                                 tolerance : float = 1e-4) -> tuple:
    '''
    P : (3, N = 10000)
    Q : (3, N = 10000)
    initial_R : (3, 3)
    initial_t : (3, 1)
    '''
    device = P.device

    P = P.t().contiguous()  # (N, 3)
    Q = Q.t().contiguous()  # (N, 3)

    R = initial_R.clone()
    t = initial_t.clone()

    kd_tree = build_kd_tree(Q)

    P_transformed = (R @ P.t() + t).t()  # (N, 3)

    for itr in range(max_iterations):
        
        dists, indices = kd_tree.query(P_transformed, nr_nns_searches=1)


        mask = dists.squeeze() < 1.0

        P_transformed_masked = P_transformed[mask]  # (N, 3)
        indices = indices[mask]  # (N, 1)
        Q_matched = Q[indices.squeeze()]  # (N, 3)

        N = P_transformed_masked.shape[0]
        e = P_transformed_masked - Q_matched  # (N, 3)
        
        I_N = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)  # (N, 3, 3)
        zeros = torch.zeros(N, device=device)
        
        skew = torch.stack([
            zeros,    -P_transformed_masked[:, 2],  P_transformed_masked[:, 1],
            P_transformed_masked[:, 2],   zeros,    -P_transformed_masked[:, 0],
        -P_transformed_masked[:, 1],  P_transformed_masked[:, 0],   zeros
        ], dim=1).reshape(N, 3, 3)
        
        J = torch.cat([I_N, -skew], dim=2)
        A = torch.bmm(J.transpose(1, 2), J).sum(dim=0)  # (6, 6)
        b = torch.bmm(J.transpose(1, 2), e.unsqueeze(-1)).sum(dim=0)  # (6, 1)

        delta = torch.linalg.lstsq(A, -b).solution  # (6, 1)
        delta_t = delta[:3]  # (3, 1)
        delta_R = exp_so3(delta[3:]) # (3, 3)

        P_transformed = (delta_R @ P_transformed.t() + delta_t).t()  # (N, 3)
        R = delta_R @ R # \hat{Q_R_P}
        t = delta_R @ t + delta_t # \hat{Q_t_P}

        if delta.norm() < tolerance:
            break
    
    return R, t

# ✅ Naive point-to-plane ICP
def po2pl(test_folder : str) -> list:
    dataset = PCD_Dataset(test_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    R_prev = torch.eye(3).to('cuda')
    t_prev = torch.zeros((3, 1)).to('cuda')

    for batch in tqdm(dataloader, desc="Test po2pl Progress"):
        P = batch['P'].to('cuda')                   # (1, 3, 10000)
        Q = batch['Q'].to('cuda')                   # (1, 3, 10000)
        R_rel_gt = batch['R_rel'].to('cuda')        # (1, 3, 3)
        t_rel_gt = batch['t_rel'].to('cuda')        # (1, 3, 1)
        P = P.squeeze(0)                             # (3, 10000)
        Q = Q.squeeze(0)                             # (3, 10000)

        R_est, t_est = icp_point2plane_gauss_newton(P, Q, initial_R = R_prev, initial_t = t_prev)

        R_prev = R_est # \hat{Q_R_P}, (3, 3)
        t_prev = t_est # \hat{Q_t_P}, (3, 1)

        R_gt = R_rel_gt.squeeze() # (3, 3)
        t_gt = t_rel_gt.squeeze(0) # (3, 1)

        results.append(((R_est.cpu(), t_est.cpu()), (R_gt.cpu(), t_gt.cpu())))
    
    return results

# ✅ Gaussian-Newton point-to-plane ICP
def icp_point2plane_gauss_newton(P : torch.tensor,
                                 Q : torch.tensor,
                                 initial_R : torch.tensor,
                                 initial_t : torch.tensor,
                                 max_iterations : int = 50,                                   
                                 tolerance : float = 1e-4) -> tuple:
    '''
    P : (3, N = 10000)
    Q : (3, N = 10000)
    initial_R : (3, 3)
    initial_t : (3, 1)
    '''
    device = P.device

    P = P.t().contiguous()  # (N, 3)
    Q = Q.t().contiguous()  # (N, 3)
    Q_plane = calcuate_plane(Q)  # (N, 3)

    R = initial_R.clone()
    t = initial_t.clone()

    kd_tree = build_kd_tree(Q)

    prev_error = math.inf

    P_transformed = (R @ P.t() + t).t()  # (N, 3)

    for itr in range(max_iterations):
        
        dists, indices = kd_tree.query(P_transformed, nr_nns_searches=1)
        mask = dists.squeeze() < 1.0
        P_transformed_masked = P_transformed[mask]  # (N, 3)
        indices = indices[mask]  # (N, 1)
        Q_matched = Q[indices.squeeze()] # (N, 3)
        Q_plane_ = Q_plane[indices.squeeze()] # (N, 3)
        
        N = P_transformed_masked.shape[0]

        e = torch.sum((P_transformed_masked - Q_matched) * Q_plane_, dim=1)  # (N,)
        J1 = Q_plane_  # (N, 3)
        zeros = torch.zeros(N, device=device)
        skew = torch.stack([
            zeros,               -P_transformed_masked[:, 2],  P_transformed_masked[:, 1],
            P_transformed_masked[:, 2],  zeros,               -P_transformed_masked[:, 0],
        -P_transformed_masked[:, 1],  P_transformed_masked[:, 0],  zeros
        ], dim=1).reshape(N, 3, 3)  # (N, 3, 3)

        J2 = -torch.bmm(Q_plane_.unsqueeze(1), skew).squeeze(1)  # (N, 3)

        J = torch.cat([J1, J2], dim=1)  # (N, 6)

        A = torch.bmm(J.unsqueeze(2), J.unsqueeze(1)).sum(dim=0)  # (6, 6)

        b = J.t() @ e.unsqueeze(1)  # (6, 1)

        delta = torch.linalg.lstsq(A, -b).solution  # (6, 1)
        delta_t = delta[:3] # (3, 1)
        delta_R = exp_so3(delta[3:]) # (3, 3)

        P_transformed = (delta_R @ P_transformed.t() + delta_t).t()  # (N, 3)
        R = delta_R @ R # \hat{Q_R_P}
        t = delta_R @ t + delta_t # \hat{Q_t_P}

        if delta.norm() < tolerance:
            break
    
    return R,t

# ✅ Naive plane-to-plane ICP
def pl2pl(test_folder : str) -> list:
    dataset = PCD_Dataset(test_folder)     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    
    R_prev = torch.eye(3).to('cuda')
    t_prev = torch.zeros((3, 1)).to('cuda')
    
    for batch in tqdm(dataloader, desc="Test pl2pl Progress"):
        P = batch['P'].to('cuda')                   # (1, 3, 10000)
        Q = batch['Q'].to('cuda')                   # (1, 3, 10000)
        R_rel_gt = batch['R_rel'].to('cuda')        # (1, 3, 3)
        t_rel_gt = batch['t_rel'].to('cuda')        # (1, 3, 1)
        C_p = calculate_gicp_cov(P.squeeze(0))
        C_q = calculate_gicp_cov(Q.squeeze(0))
        #! sometimes the cuda assertion error is due to the inital rotation? when it is wrong, it worsens transformation and makes kdtree query fail?
        R_est, t_est = icp_generalized_gauss_newton(P.squeeze(0), Q.squeeze(0), R_prev, t_prev, C_p, C_q)

        R_prev = R_est.clone() # \hat{Q_R_P}, (3, 3)
        t_prev = t_est.clone() # \hat{Q_t_P}, (3, 1)

        R_gt= R_rel_gt.squeeze(0)
        t_gt= t_rel_gt.squeeze(0)

        results.append(((R_est.cpu(), t_est.cpu()), (R_gt.cpu(), t_gt.cpu())))

    return results

# ✅ Gaussian-Newton plane-to-plane ICP
def icp_generalized_gauss_newton_corr(P_masked: torch.tensor,
                                 Q_matched: torch.tensor,
                                 information_masked: torch.tensor,) -> tuple:
    device = P_masked.device

    N = P_masked.shape[0]
    e = P_masked - Q_matched  # [M, 3]

    I_N = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)  # [M, 3, 3]
    zeros = torch.zeros(N, device=device)  # [M]

    skew = torch.stack([
        zeros, -P_masked[:, 2], P_masked[:, 1],
        P_masked[:, 2], zeros, -P_masked[:, 0],
        -P_masked[:, 1], P_masked[:, 0], zeros
    ], dim=1).reshape(N, 3, 3)  # [M, 3, 3]

    J = torch.cat([I_N, -skew], dim=2)  # [M, 3, 6]
    WJ = torch.bmm(information_masked, J)  # [M, 3, 6]
    JTWJ = torch.bmm(J.transpose(1, 2), WJ)  # [M, 6, 6]
    Wy = torch.bmm(information_masked, e.unsqueeze(-1))  # [M, 3, 1]
    JTWy = torch.bmm(J.transpose(1, 2), Wy)  # [M, 6, 1]
    JTWJ = JTWJ.mean(dim=0)  # [6, 6]
    JTWy = JTWy.mean(dim=0)  # [6, 1]

    delta = torch.linalg.lstsq(JTWJ, -JTWy).solution  # [6, 1]
    delta_t = delta[:3]  # [3, 1]
    delta_R = exp_so3(delta[3:])  # [3, 3]
    # delta_R = exp_so3(delta[3:].view(-1))

    return delta_R, delta_t

def exp_so3_batch(omega: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix exponential of batched so(3) vectors.
    omega: [B, 3]
    return: [B, 3, 3]
    """
    B = omega.shape[0]
    device = omega.device
    theta = torch.norm(omega, dim=1, keepdim=True)  # [B, 1]
    theta = torch.clamp(theta, min=1e-8)
    omega_unit = omega / theta  # [B, 3]

    x, y, z = omega_unit[:, 0], omega_unit[:, 1], omega_unit[:, 2]
    zeros = torch.zeros(B, device=device)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=1).reshape(B, 3, 3)

    I = torch.eye(3, device=device).expand(B, 3, 3)
    theta = theta.view(B, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.bmm(K, K)
    return R

def icp_generalized_gauss_newton_corr_batch(
    P_masked: torch.Tensor,             # [B, M, 3]
    Q_matched: torch.Tensor,           # [B, M, 3]
    information_masked: torch.Tensor,  # [B, M, 3, 3]
    max_iterations: int = 50,
    tolerance: float = 1e-4
) -> tuple:
    B, N, _ = P_masked.shape
    device = P_masked.device

    e = P_masked - Q_matched  # [B, M, 3]

    # Identity and skew matrices
    I_N = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3)  # [B, M, 3, 3]
    zeros = torch.zeros((B, N), device=device)  # [B, M]

    px, py, pz = P_masked[:, :, 0], P_masked[:, :, 1], P_masked[:, :, 2]
    skew = torch.stack([
        zeros, -pz, py,
        pz, zeros, -px,
        -py, px, zeros
    ], dim=-1).reshape(B, N, 3, 3)  # [B, M, 3, 3]

    # Jacobian: [B, M, 3, 6]
    J = torch.cat([I_N, -skew], dim=-1)  # [B, M, 3, 6]

    # WJ = information @ J : [B, M, 3, 6]
    WJ = torch.matmul(information_masked, J)  # [B, M, 3, 6]

    # J^T W J : [B, M, 6, 6]
    J_t = J.transpose(-2, -1)  # [B, M, 6, 3]
    JTWJ = torch.matmul(J_t, WJ)  # [B, M, 6, 6]

    # Wy = information @ e : [B, M, 3, 1]
    Wy = torch.matmul(information_masked, e.unsqueeze(-1))  # [B, M, 3, 1]

    # J^T W y : [B, M, 6, 1]
    JTWy = torch.matmul(J_t, Wy)  # [B, M, 6, 1]

    # Average over M points
    JTWJ_mean = JTWJ.mean(dim=1)  # [B, 6, 6]
    JTWy_mean = JTWy.mean(dim=1)  # [B, 6, 1]

    # Solve (J^T W J) delta = - J^T W y
    delta = torch.linalg.lstsq(JTWJ_mean, -JTWy_mean).solution  # [B, 6, 1]
    delta_t = delta[:, :3]  # [B, 3, 1]
    delta_w = delta[:, 3:]  # [B, 3, 1]

    # Exponential map to get rotation matrices
    delta_R = exp_so3_batch(delta_w.squeeze(-1))  # [B, 3, 3]

    return delta_R, delta_t  # [B, 3, 3], [B, 3, 1]

def icp_generalized_gauss_newton_corr_batch_softmask(
    P_masked: torch.Tensor,             # [B, M, 3]
    Q_matched: torch.Tensor,           # [B, M, 3]
    information_masked: torch.Tensor,  # [B, M, 3, 3]
    mask: torch.Tensor,                # [B, M]
) -> tuple:
    B, N, _ = P_masked.shape
    device = P_masked.device

    e = P_masked - Q_matched  # [B, M, 3]

    # Identity and skew matrices
    I_N = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3)  # [B, M, 3, 3]
    zeros = torch.zeros((B, N), device=device)  # [B, M]

    px, py, pz = P_masked[:, :, 0], P_masked[:, :, 1], P_masked[:, :, 2]
    skew = torch.stack([
        zeros, -pz, py,
        pz, zeros, -px,
        -py, px, zeros
    ], dim=-1).reshape(B, N, 3, 3)  # [B, M, 3, 3]

    # Jacobian: [B, M, 3, 6]
    J = torch.cat([I_N, -skew], dim=-1)  # [B, M, 3, 6]

    # WJ = information @ J : [B, M, 3, 6]
    WJ = torch.matmul(information_masked, J)  # [B, M, 3, 6]

    # J^T W J : [B, M, 6, 6]
    J_t = J.transpose(-2, -1)  # [B, M, 6, 3]
    JTWJ = torch.matmul(J_t, WJ)  # [B, M, 6, 6]

    # Wy = information @ e : [B, M, 3, 1]
    Wy = torch.matmul(information_masked, e.unsqueeze(-1))  # [B, M, 3, 1]

    # J^T W y : [B, M, 6, 1]
    JTWy = torch.matmul(J_t, Wy)  # [B, M, 6, 1]

    # Average over M points
    JTWJ = JTWJ * mask.unsqueeze(-1).unsqueeze(-1)  # [B, M, 6, 6]
    JTWy = JTWy * mask.unsqueeze(-1).unsqueeze(-1)  # [B, M, 6, 1]
    JTWJ_sum = JTWJ.sum(dim=1)  # [B, 6, 6]
    JTWy_sum = JTWy.sum(dim=1)  # [B, 6, 1]

    # Solve (J^T W J) delta = - J^T W y
    delta = torch.linalg.lstsq(JTWJ_sum, -JTWy_sum).solution  # [B, 6, 1]
    delta_t = delta[:, :3]  # [B, 3, 1]
    delta_w = delta[:, 3:]  # [B, 3, 1]

    # Exponential map to get rotation matrices
    delta_R = exp_so3_batch(delta_w.squeeze(-1))  # [B, 3, 3]

    return delta_R, delta_t  # [B, 3, 3], [B, 3, 1]

def find_nearest_neighbors(src: torch.Tensor, dst: torch.Tensor) -> tuple:
    """
    src: (N, 3) transformed source points (P')
    dst: (M, 3) target points (Q)
    
    returns:
        dists: (N,) 가장 가까운 점까지의 거리
        indices: (N,) 가장 가까운 점의 인덱스
    """
    # 거리 계산 (브로드캐스팅 없이, GPU 가능)
    dists = torch.cdist(src.unsqueeze(0), dst.unsqueeze(0)).squeeze(0)  # [N, M]
    min_dists, indices = dists.min(dim=1)  # 각 P_i에 대해 가장 가까운 Q_j
    return min_dists, indices

# ✅ Gaussian-Newton plane-to-plane ICP
def icp_generalized_gauss_newton(P: torch.tensor,
                                 Q: torch.tensor,
                                 initial_R: torch.tensor,
                                 initial_t: torch.tensor,
                                 C_p: torch.tensor,
                                 C_q: torch.tensor,
                                 max_iterations: int = 50,
                                 tolerance: float = 1e-4) -> tuple:
    epsilon = torch.eye(3).expand(1, P.shape[1], 3, 3).to('cuda')*1e-3
    device = P.device
    P = P.t().contiguous().float()  # [N, 3], float32로 변환
    Q = Q.t().contiguous().float()  # [N, 3], float32로 변환
    R = initial_R.clone()   # [3, 3]
    t = initial_t.clone()   # [3, 1]
    P_transformed = (R @ P.t() + t).t()  # [N, 3]
    # kd_tree = build_kd_tree(Q)
    for itr in range(max_iterations):
        # dists, indices = kd_tree.query(P_transformed, nr_nns_searches=1)
        dists, indices = find_nearest_neighbors(P_transformed, Q)  # [N], [N]
        indices = indices.squeeze()  # [N]
        
        C_q_matched = C_q[indices]  # [N, 3, 3]
        information_matrix = torch.inverse(C_q_matched + C_p+ epsilon.squeeze(0))  # [N, 3, 3]

        mask = dists.squeeze() < 1.0  # [N]
        P_transformed_masked = P_transformed[mask]  # [M, 3]
        indices_masked = indices[mask]  # [M]
        Q_matched = Q[indices_masked]  # [M, 3]
        information_matrix_masked = information_matrix[mask]  # [M, 3, 3]
        N = P_transformed_masked.shape[0]
        e = P_transformed_masked - Q_matched  # [M, 3]

        I_N = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)  # [M, 3, 3]
        zeros = torch.zeros(N, device=device)  # [M]

        skew = torch.stack([
            zeros, -P_transformed_masked[:, 2], P_transformed_masked[:, 1],
            P_transformed_masked[:, 2], zeros, -P_transformed_masked[:, 0],
            -P_transformed_masked[:, 1], P_transformed_masked[:, 0], zeros
        ], dim=1).reshape(N, 3, 3)  # [M, 3, 3]

        J = torch.cat([I_N, -skew], dim=2)  # [M, 3, 6]
        WJ = torch.bmm(information_matrix_masked, J)  # [M, 3, 6]
        JTWJ = torch.bmm(J.transpose(1, 2), WJ)  # [M, 6, 6]
        Wy = torch.bmm(information_matrix_masked, e.unsqueeze(-1))  # [M, 3, 1]
        JTWy = torch.bmm(J.transpose(1, 2), Wy)  # [M, 6, 1]
        JTWJ = JTWJ.mean(dim=0)  # [6, 6]
        JTWy = JTWy.mean(dim=0)  # [6, 1]

        delta = torch.linalg.lstsq(JTWJ, -JTWy).solution  # [6, 1]
        delta_t = delta[:3]  # [3, 1]
        delta_R = exp_so3(delta[3:])  # [3, 3]

        P_transformed = (delta_R @ P_transformed.t() + delta_t).t()  # [N, 3]
        R = delta_R @ R
        t = delta_R @ t + delta_t
        C_p= delta_R @ C_p @ delta_R.t()

        if delta.norm() < tolerance:
            break

    return R, t

# ✅ Calculate plane from target point cloud
def calcuate_plane(Q : torch.tensor) -> torch.tensor:
    '''
    Q : (N, 3)
    return : (N, 3)
    '''
    device = Q.device
    N = Q.shape[0]

    kd_tree = build_kd_tree(Q)

    dists, idxs = kd_tree.query(Q, nr_nns_searches=10)
    neighbors = Q[idxs]  # (N, 10, 3)

    mean_local = neighbors.mean(dim=1)  # (N, 3)
    X = neighbors - mean_local.unsqueeze(1)  # (N, 10, 3)
    cov = torch.bmm(X.transpose(1,2), X)/10  # (N, 3, 3)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # (N, 3), (N, 3, 3)
    normals = eigvecs[:, :, 0]  # (N, 3)
    normals = normals / torch.norm(normals, dim=1, keepdim=True).view(-1, 1)  # (N, 3)

    return normals

def calculate_gicp_cov(Q: torch.tensor) -> torch.tensor:
    '''
    Q : (N, 3)
    return : normals (N, 3), eigvecs (N, 3, 3)
    '''
    Q= Q.squeeze(0).t()
    N= Q.shape[0]
    eps= 1e-6
    # 1. KD-tree를 사용해 최근접 이웃 찾기
    kd_tree = build_kd_tree(Q)
    dists, idxs = kd_tree.query(Q, nr_nns_searches=10)
    neighbors = Q[idxs]  # (N, nr_nns_searches, 3)

    # 2. 로컬 이웃의 평균 계산
    mean_local = neighbors.mean(dim=1)  # (N, 3)

    # 3. 편차 행렬 구성
    X = neighbors - mean_local.unsqueeze(1)  # (N, nr_nns_searches, 3)

    # 4. 공분산 행렬 계산
    cov = torch.bmm(X.transpose(1, 2), X) / 10  # (N, 3, 3)

    # 5. 수치적 안정성을 위해 작은 값 추가
    I = torch.eye(3, device=Q.device).unsqueeze(0).expand(N, 3, 3)  # (N, 3, 3)
    cov = cov + eps * I

    return cov

def rotvec_to_rotmat(rvec):
    # rvec의 shape: [B, 3]
    B = rvec.shape[0]
    theta = torch.norm(rvec, dim=1, keepdim=True)  # [B, 1]
    
    # 회전 행렬 R를 [B, 3, 3] 크기로 초기화
    R = torch.zeros(B, 3, 3, device=rvec.device, dtype=rvec.dtype)
    
    # theta가 작은 경우의 mask (회전각이 0에 가까우면 identity 반환)
    mask = (theta < 1e-6).squeeze(1)  # [B]
    
    # Identity 행렬을 배치에 맞게 확장
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0).expand(B, -1, -1)
    
    # 작은 각도에 대해서는 identity 할당
    R[mask] = I[mask]
    
    # 작은 각도가 아닌 경우에 대해 Rodrigues 공식을 사용
    if (~mask).sum() > 0:
        rvec_nonzero = rvec[~mask]      # [B_nonzero, 3]
        theta_nonzero = theta[~mask]    # [B_nonzero, 1]
        u = rvec_nonzero / theta_nonzero  # unit vector, [B_nonzero, 3]
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        # skew-symmetric matrix K
        K = torch.stack([
            torch.zeros_like(ux), -uz, uy,
            uz, torch.zeros_like(ux), -ux,
            -uy, ux, torch.zeros_like(ux)
        ], dim=1).view(-1, 3, 3)  # [B_nonzero, 3, 3]
        
        # Rodrigues 공식: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        R_nonzero = I[~mask] + torch.sin(theta_nonzero) * K + (1 - torch.cos(theta_nonzero)) * (K @ K)
        R[~mask] = R_nonzero
        
    return R  # 이미 rvec와 같은 device에 있으므로, .cuda()는 생략 가능

def informed_gaussian_newton(P : torch.tensor,
                             Q : torch.tensor,
                            information_matrix : torch.tensor,
                            initial_R : torch.tensor,
                            initial_t : torch.tensor,
                            max_iterations : int = 100,
                            tolerance : float = 1e-5) -> tuple:
    '''
    P : (3, N = 10000)
    Q : (3, N = 10000)
    information_matrix : (N, 3, 3)
    initial_R : (3, 3)
    initial_t : (3, 1)
    '''

    device = P.device

    P = P.t().contiguous()  # (N, 3)
    Q = Q.t().contiguous()  # (N, 3)
    R = initial_R.clone()
    t = initial_t.clone()

    kd_tree = build_kd_tree(Q)

    P_transformed = (R @ P.t() + t).t()  # (N, 3)
    
    for itr in range(max_iterations):
        
        dists, indices = kd_tree.query(P_transformed, nr_nns_searches=1)

        mask = dists.squeeze() < 1 # (N,)

        P_transformed_masked = P_transformed[mask]  # (N, 3)
        indices = indices[mask] # (N, 1)
        Q_matched = Q[indices.squeeze()] # (N, 3)
        information_matrix_masked = information_matrix[mask] # (N, 3, 3)

        N = P_transformed_masked.shape[0]
        e = P_transformed_masked - Q_matched # (N, 3)

        '''
        Weighted Least Square
        W = information_matrix_masked : (N, 3, 3)
        y = e : (N, 3)
        X = J : (N, 3, 6)
        x = delta : (6, 1)
        '''

        I_N = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)  # (N, 3, 3)
        zeros = torch.zeros(N, device=device) # (N,)

        skew = torch.stack([
            zeros,    -P_transformed_masked[:, 2],  P_transformed_masked[:, 1],
            P_transformed_masked[:, 2],   zeros,    -P_transformed_masked[:, 0],
        -P_transformed_masked[:, 1],  P_transformed_masked[:, 0],   zeros
        ], dim=1).reshape(N, 3, 3) # (N, 3, 3)

        J = torch.cat([I_N, -skew], dim=2) # (N, 3, 6)
        WJ = torch.bmm(information_matrix_masked, J) # (N, 3, 6)
        JTWJ = torch.bmm(J.transpose(1, 2), WJ) # (N, 6, 6)
        Wy = torch.bmm(information_matrix_masked, e.unsqueeze(-1)) # (N, 3, 1)
        JTWy = torch.bmm(J.transpose(1, 2), Wy) # (N, 6, 1)
        JTWJ = JTWJ.sum(dim=0)  # (6, 6)
        JTWy = JTWy.sum(dim=0) # (6, 1)

        delta = torch.linalg.lstsq(JTWJ, -JTWy).solution  # (6, 1)
        delta_t = delta[:3] # (3, 1)
        delta_R = exp_so3(delta[3:]) # (3, 3)

        P_transformed = (delta_R @ P_transformed.t() + delta_t).t()  # (N, 3)
        R = delta_R @ R
        t = delta_R @ t + delta_t

        if delta.norm() < tolerance:
            break

    return R, t

# ✅ Tools 
def project_point_cloud(points, resolution=(32, 512), f_h=360.0, f_vu=11.25, f_vl=-11.25):
    device = points.device
    height, width = resolution

    f_h_rad = math.radians(f_h)
    f_vu_rad = math.radians(f_vu)
    f_vl_rad = math.radians(f_vl)
    f_v_rad = f_vu_rad - f_vl_rad

    delta_h = f_h_rad / width
    delta_v = f_v_rad / height

    x, y, z = points[:,0], points[:,1], points[:,2]
    ranges = torch.sqrt(x**2 + y**2 + z**2)
    d = torch.sqrt(x**2 + y**2 + 1e-6)

    theta_h = torch.atan2(y, x)
    theta_v = torch.atan2(z, d)

    u = (f_h_rad/2 - theta_h) / delta_h
    v = (f_vu_rad - theta_v) / delta_v

    u_idx = torch.floor(u).long() % width
    v_idx = torch.floor(v).long().clamp(0, height - 1)
    indices = v_idx * width + u_idx

    depth_image_flat = torch.full((height * width,), float('inf'), device=device)
    try:
        depth_image_flat = depth_image_flat.scatter_reduce(0, indices, ranges, reduce='amin', include_self=True)
    except AttributeError:
        for i in range(points.shape[0]):
            idx = indices[i].item()
            depth_image_flat[idx] = min(depth_image_flat[idx].item(), ranges[i].item())

    depth_image = depth_image_flat.view(height, width)
    depth_image[depth_image == float('inf')] = 0

    max_depth = depth_image.max()
    if max_depth > 0:
        depth_image_normalized = 255 * (1 - depth_image / max_depth)
    else:
        depth_image_normalized = depth_image

    return depth_image_normalized.to(torch.uint8)

def project_point_cloud_with_values(points, values, resolution=(32, 512), f_h=360.0, f_vu=11.25, f_vl=-11.25):
    device = points.device
    height, width = resolution

    f_h_rad = math.radians(f_h)
    f_vu_rad = math.radians(f_vu)
    f_vl_rad = math.radians(f_vl)
    f_v_rad = f_vu_rad - f_vl_rad

    delta_h = f_h_rad / width
    delta_v = f_v_rad / height

    x, y, z = points[:,0], points[:,1], points[:,2]
    d = torch.sqrt(x**2 + y**2 + 1e-6)
    theta_h = torch.atan2(y, x)
    theta_v = torch.atan2(z, d)

    u = (f_h_rad/2 - theta_h) / delta_h
    v = (f_vu_rad - theta_v) / delta_v

    u_idx = torch.floor(u).long() % width
    v_idx = torch.floor(v).long().clamp(0, height - 1)
    indices = v_idx * width + u_idx

    image_flat = torch.full((height * width,), float('inf'), device=device)
    try:
        image_flat = image_flat.scatter_reduce(0, indices, values, reduce='amin', include_self=True)
    except AttributeError:
        for i in range(points.shape[0]):
            idx = indices[i].item()
            image_flat[idx] = min(image_flat[idx].item(), values[i].item())

    image = image_flat.view(height, width)
    image[image == float('inf')] = 0

    max_val = image.max()
    if max_val > 0:
        image_normalized = 255 * (1 - image / max_val)
    else:
        image_normalized = image

    return image_normalized.to(torch.uint8)

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    v: (3,) shape
    return: [v]_x, (3,3) shape
    """
    return torch.tensor([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],    0]
    ], dtype=v.dtype, device=v.device)

def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """
    w: (3,) shape, 회전 벡터(미소 회전)
    Rodrigues 공식으로부터 exp([w]_x) 계산
    """
    theta = torch.norm(w)
    if theta < 1e-12:
        return torch.eye(3, device=w.device, dtype=w.dtype)
    w_unit = w / theta
    K = skew_symmetric(w_unit)
    return (
        torch.eye(3, device=w.device, dtype=w.dtype)
        + torch.sin(theta)*K
        + (1 - torch.cos(theta))*(K @ K)
    )




def sibal_corr_icp_testing(test_folder : str) -> list:
    dataset = PCD_Dataset(test_folder)     
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    
    for batch in tqdm(dataloader, desc="Test corr_icp Progress"):
        P = batch['P'].to('cuda')                   # (1, 3, 10000)
        Q = batch['Q'].to('cuda')                   # (1, 3, 10000)
        R_rel_gt = batch['R_rel'].to('cuda')        # (1, 3, 3)
        t_rel_gt = batch['t_rel'].to('cuda')        # (1, 3, 1)
        indices = batch['indices'].to('cuda')          # (1, 10000)
        P_masked = P.squeeze(0).t()  # (3, N)
        Q_matched = Q[0, :, indices.squeeze(0)].t()
        # information matrix as identity
        information = torch.eye(3, device='cuda').repeat(5000, 1, 1)
        #! sometimes the cuda assertion error is due to the inital rotation? when it is wrong, it worsens transformation and makes kdtree query fail?
        R_est, t_est = icp_generalized_gauss_newton_corr(P_masked, Q_matched, information)

        R_gt= R_rel_gt.squeeze(0)
        t_gt= t_rel_gt.squeeze(0)

        results.append(((R_est.cpu(), t_est.cpu()), (R_gt.cpu(), t_gt.cpu())))

    return results
