import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import random
# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})

def estimate_homo(kpts0, kpts1, M, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    #E, mask = cv2.findEssentialMat(
    #    kpts0, kpts1, np.eye(3), prob=conf, method=None)

    #E, mask = cv2.findEssentialMat(
    #    kpts0, kpts1, np.eye(3), prob=conf, method=None)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def compute_homo_errors(data, config):
    """
    Update:
        data (dict):{
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'inliers': []})
    data.update({'epi_errs': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    M=data['M']
    #print(data)
    #K0 = data['K0'].cpu().numpy()
    #K1 = data['K1'].cpu().numpy()
    #T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(data['image0'].shape[0]):
        mask = m_bids == bs
        kpts0 = pts0[mask]
        kpts1 = pts1[mask]
        M_b = M[bs]
        if kpts0.shape[0]==0:
            data['inliers'].append(np.array([]).astype(np.bool))
            data['epi_errs'].append(np.array([]).astype(np.bool))
        else:
            kpts0 = kpts0.reshape((1, -1, 2))
            kpts0 = cv2.perspectiveTransform(kpts0, M_b.cpu().numpy())
            inliers=0
            epi_errs = []
            for ii,cord in enumerate(kpts0[0]):
                diff = cord-kpts1[ii]
                if (diff[0]**2+diff[1]**2)<=4:
                    inliers+=1
                epi_errs.append(np.sqrt(diff[0]**2+diff[1]**2))
            data['epi_errs'].append(np.array(epi_errs))
            data['inliers'].append(inliers)

def filter_based_on_depth(depth0,depth1,coordinates0,coordinates1,K0,K1,T_0to1):
    coordinates0=coordinates0[None,...]
    coordinates1 = coordinates1[None, ...]
    coordinates0 =coordinates0.long()
    coordinates1 =coordinates1.long()
    kpts0_depth = torch.stack([depth0[coordinates0[0,:, 1], coordinates0[0,:, 0]]], dim=0)
    nonzero_mask = (kpts0_depth != 0)*float('inf')
    kpts0_h = torch.cat([coordinates0, torch.ones_like(coordinates0[:, :, [0]])], dim=-1) * kpts0_depth[
        ..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    # Rigid Transform
    w_kpts0_cam = T_0to1[:3, :3] @ kpts0_cam + T_0to1[:3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[0:2]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w - 1) * \
                     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h - 1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack([depth1[coordinates1[0, :, 1], coordinates1[0, :, 0]]], dim=0)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    #diff = (abs(w_kpts0_depth - w_kpts0_depth_computed)/(w_kpts0_depth+1e-4))
    diff = abs((w_kpts0_depth - w_kpts0_depth_computed)/(w_kpts0_depth+1e-4))
    #diff *= nonzero_mask
    indice = torch.where(diff>0.15)
    #print(diff.size())
    #print(len(indice[1]))
    new_cor0 = coordinates0[indice[0],indice[1]]
    new_cor1 = coordinates1[indice[0],indice[1]]
    return indice[1]#new_cor0,new_cor1

def filter_depth_inconsist_point(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu()#.numpy()
    pts1 = data['mkpts1_f'].cpu()#.numpy()
    depth0 = data['depth0'].cpu()#.numpy()
    depth1 = data['depth1'].cpu()#.numpy()# shape (1,480,640)
    K0 = data['K0'].cpu()#.numpy()
    K1 = data['K1'].cpu()#.numpy()
    T_0to1 = data['T_0to1'].cpu()#.numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        #ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)
        ind=filter_based_on_depth(depth0[bs],depth1[bs],pts0, pts1, K0[bs], K1[bs],T_0to1[bs])

        m_bids_new = data['m_bids']
        m_bids_new[ind]=-1
        data.update({'m_bids': m_bids_new.cuda()})
        #data.update({'mkpts0_f': new_cor0.cuda(), 'mkpts1_f': new_cor1.cuda(),'m_bids': m_bids_new.cuda()})
        m_bids = data['m_bids'].cpu().numpy()
        pts0 = data['mkpts0_f'].cpu().numpy()
        mask = m_bids == bs
        pts1 = data['mkpts1_f'].cpu().numpy()
        K0 = data['K0'].cpu().numpy()
        K1 = data['K1'].cpu().numpy()
        T_0to1 = data['T_0to1'].cpu().numpy()
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)
        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)

def filter_based_random_sample(depth0,depth1,pts0, pts1):
    max_depth = depth0.max()
    h, w = depth0.shape[0:2]
    scale =8
    h = h//8
    w = w//8
    uni_pb = 1./float(h*w*10000)
    total = pts0.size(0)
    rest = 1 - uni_pb*total
    set_ind = np.arange(total+1)
    pb_ind = [uni_pb]*total+[rest]
    np.random.seed()
    ind = np.random.choice(set_ind,size = (int(total/5)),replace=False, p = pb_ind)
    dust_bin = np.where(ind==total)[0]
    try:
        ind =list(ind)
        ind.pop(dust_bin[0])
        return ind
    except:
        return ind






def filter_unsampled_point(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu()#.numpy()
    pts1 = data['mkpts1_f'].cpu()#.numpy()
    depth0 = data['depth0'].cpu()#.numpy()
    depth1 = data['depth1'].cpu()#.numpy()# shape (1,480,640)
    K0 = data['K0'].cpu()#.numpy()
    K1 = data['K1'].cpu()#.numpy()
    T_0to1 = data['T_0to1'].cpu()#.numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        #ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)
        ind=filter_based_random_sample(depth0[bs],depth1[bs],pts0, pts1)

        m_bids_new = data['m_bids']
        m_bids_new[ind]=-1
        data.update({'m_bids': m_bids_new.cuda()})
        #data.update({'mkpts0_f': new_cor0.cuda(), 'mkpts1_f': new_cor1.cuda(),'m_bids': m_bids_new.cuda()})
        m_bids = data['m_bids'].cpu().numpy()
        pts0 = data['mkpts0_f'].cpu().numpy()
        mask = m_bids == bs
        pts1 = data['mkpts1_f'].cpu().numpy()
        K0 = data['K0'].cpu().numpy()
        K1 = data['K1'].cpu().numpy()
        T_0to1 = data['T_0to1'].cpu().numpy()
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)
        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)

def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs}

def aggregate_metrics_homo(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    #unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    #unq_ids = list(unq_ids.values())
    #logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    #angular_thresholds = [5, 10, 20]
    #pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)#[unq_ids]
    #aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object), dist_thresholds, True)  # (prec@err_thr)

    return { **precs}