"""
Segmentation tools
@Author: Yijing Yang
@Contact: yangyijing710@outlook.com
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def resize_feat(feat, target, interp='lanczos'):
    feat_resized = np.zeros((feat.shape[0], target[0], target[1], feat.shape[-1]))
    for n in range(feat.shape[0]):
        for c in range(feat.shape[-1]):
            if interp == 'lanczos':
                feat_resized[n,:,:,c] = cv2.resize(feat[n,:,:,c], (target[0], target[1]), interpolation=cv2.INTER_LANCZOS4)
            elif interp == 'bicubic':
                feat_resized[n,:,:,c] = cv2.resize(feat[n,:,:,c], (target[0], target[1]), interpolation=cv2.INTER_CUBIC)
            elif interp == 'bilinear':
                feat_resized[n,:,:,c] = cv2.resize(feat[n,:,:,c], (target[0], target[1]), interpolation=cv2.INTER_LINEAR)
    return feat_resized

# def up_stream(down_feat, num_layer=2,rate=2, interp='lanczos'): # 'bicubic', 'bilinear', 'lanczos'
#     total_layer = len(down_feat)
#     # assert (target_dim[0]//down_feat[-1].shape[1]) == (target_dim[1]//down_feat[-1].shape[2])
#     for i in range(num_layer):
#         if i==0:
#             merged_feat = np.copy(down_feat[-1])
#         else:
#             _, H, W, _ = merged_feat.shape
#             upsampled_feat = resize_feat(merged_feat, [H * rate, W * rate], interp=interp)
#             print(upsampled_feat.shape)
#             merged_feat = np.concatenate((down_feat[total_layer-1-i],upsampled_feat),axis=-1)
#
#     return merged_feat


def up_stream(down_feat, num_layer=2,target=32, interp='bilinear'): # 'bicubic', 'bilinear', 'lanczos'
    total_layer = len(down_feat)
    # assert (target_dim[0]//down_feat[-1].shape[1]) == (target_dim[1]//down_feat[-1].shape[2])
    total_feat_dim = 0
    total_feat_dim_each = []
    for i in range(num_layer):
        total_feat_dim += down_feat[i].shape[-1]
        total_feat_dim_each.append(down_feat[i].shape[-1])
    total_feat_dim_each = np.array(total_feat_dim_each)
    total_feat_dim_each = np.cumsum(total_feat_dim_each)
    upsampled_feat = np.zeros((down_feat[0].shape[0], target, target, total_feat_dim))

    for i in range(num_layer):
        if i==0:
            upsampled_feat[:,:,:,:total_feat_dim_each[i]] = down_feat[i]
            # tmp = resize_feat(down_feat[i], [target,target], interp=interp)
            # print(tmp.shape)
            # upsampled_feat[:,:,:,:total_feat_dim_each[i]] = tmp
            # del tmp
        else:
            tmp = resize_feat(down_feat[i], [target,target], interp=interp)
            print(tmp.shape)
            upsampled_feat[:,:,:,total_feat_dim_each[i-1]:total_feat_dim_each[i]] = tmp
            del tmp
        down_feat[i] = []

    return upsampled_feat

def down_stream(down_feat, START=None,total_layer=3,rate=0.5, interp='lanczos'): # 'bicubic', 'bilinear', 'lanczos'
    # num_layer = len(down_feat)
    # assert (target_dim[0]//down_feat[-1].shape[1]) == (target_dim[1]//down_feat[-1].shape[2])

    for i in range(START,total_layer):
        if i==START:
            merged_feat = np.copy(down_feat[i])
        else:
            _, H, W, _ = merged_feat.shape
            downsampled_feat = resize_feat(merged_feat, [int(H * rate), int(W * rate)], interp=interp)
            print(downsampled_feat.shape)
            merged_feat = np.concatenate((downsampled_feat,down_feat[i]),axis=-1)

    return merged_feat