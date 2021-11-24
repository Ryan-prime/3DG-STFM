import os
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors

from src.loftr import default_cfg, LoFTR_RGBD, LoFTR_RGB
import matplotlib.pyplot as plt


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c='b', linewidth=0.5,alpha=0.3)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=5)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=5)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color='k')

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        print('saved', os.getcwd(), path)
        plt.close()
    else:
        return fig


def pose_filter(mkpts0, mkpts1, K0, K1):
    mkpts0 = (mkpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    mkpts1 = (mkpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    ransac_thr = 0.5 / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    if len(mkpts0) < 6:
        E = None
        mask = None
    else:
        E, mask = cv2.findEssentialMat(
            mkpts0, mkpts1, np.eye(3), threshold=ransac_thr, prob=0.99999, method=cv2.RANSAC)
    return E, mask

root_dir = 'inference/'
pretrained_ckpt = "weights/indoor_student.ckpt"
matcher = LoFTR_RGB(config=default_cfg)
img0_pth, img1_pth = 'demo1.jpg','demo2.jpg'

img0_pth, img1_pth = root_dir + img0_pth, root_dir + img1_pth
sd = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in sd.items():
    name = k[8:]  # remove `matcher.`
    new_state_dict[name] = v
matcher.load_state_dict(new_state_dict, strict=False)
matcher = matcher.eval().cuda()

img0_raw = cv2.imread(img0_pth, cv2.IMREAD_COLOR)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_COLOR)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB)

img0 = np.ascontiguousarray(img0)
img1 = np.ascontiguousarray(img1)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
img0 = img0.astype(float)
img1 = img1.astype(float)
img0[:, :, 0] = (img0[:, :, 0] / 255. - mean[0]) / std[0]
img0[:, :, 1] = (img0[:, :, 1] / 255. - mean[1]) / std[1]
img0[:, :, 2] = (img0[:, :, 2] / 255. - mean[2]) / std[2]
img1[:, :, 0] = (img1[:, :, 0] / 255. - mean[0]) / std[0]
img1[:, :, 1] = (img1[:, :, 1] / 255. - mean[1]) / std[1]
img1[:, :, 2] = (img1[:, :, 2] / 255. - mean[2]) / std[2]

img0 = torch.from_numpy(img0).float()[None].cuda()
img1 = torch.from_numpy(img1).float()[None].cuda()
img0 = img0.permute(0, 3, 1, 2)
img1 = img1.permute(0, 3, 1, 2)
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

#_, mask = pose_filter(mkpts0, mkpts1, K0, K1)
# ind_mask = np.where(mask == 1)
# mkpts0 = mkpts0[ind_mask[0], :]
# mkpts1 = mkpts1[ind_mask[0], :]
# mconf = mconf[ind_mask[0]]

# Draw
if mconf!=[]:
    mconf=(mconf-mconf.min())/(mconf.max()-mconf.min())
color = cm.jet(mconf)
text = [
    '3DG-STFM',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text,
                           path='demo.png')
