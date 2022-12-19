#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:17:55 2019
@author: Wei
"""


import math
import random
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#Part 0: check data quality, set/import hyper-parameters  # module part1
#Part 1: import dataset
#Part 2: split training-validation dataset
#Part 3: get samples (images -> patches)
#Part 4: balance samples
#Part 5: extract feature and do normalization  # model part2
#Part 6: train/apply classifiers  # model part3
#Part 7: ensemble results


#%%
# basics, fundamental functions
def wipe_surrounding(img, bbox_corner_pair):
    '''
    wipe defect proposals in one image outside bboxes
    @ Args:
        img(np.array): [h,w]
        bbox_corner_pair(np.array): [[h0,w0], [h1,w1]],
            point0: upper-left point, point1: bottom-right point
    '''
#    h,w = img.shape
    img[:bbox_corner_pair[0][0], :] = 0
    img[bbox_corner_pair[1][0]:, :] = 0
    img[:, :bbox_corner_pair[0][1]] = 0
    img[:, bbox_corner_pair[1][1]:] = 0
    return img


def cropping(patch, bbox_corner_pair):
    '''
    return patch cropped by bbox
    @ Args:
        img(np.array): [h,w]
        bbox_corner_pair(np.array): [[h0,w0], [h1,w1]],
                                    point0: upper-left point,
                                    point1: bottom-right point
    '''
    return patch[bbox_corner_pair[0][0]:bbox_corner_pair[1][0],
                 bbox_corner_pair[0][1]:bbox_corner_pair[1][1]]


def regulize_point(point):
    '''
    when point is outside the image, regulize its coordinate to image border
    @ Args:
        point(2d tuple): e.g. (h,w)
    '''
    row = max(point[0], 0)
    row = min(row, 479)
    col = max(point[1], 0)
    col = min(col, 479)
    return (row, col)


# detection Module 1.1
def remove_regular_pattern_auto(input_img):
    '''
    funtion: detect potential defect area for a single image
    @input:
        input_img(2D np.array, uint8): one defect image, [image_height, image_width]
    @output:
        edge_map_defect(2D np.array, uint8): [image_height, image_width],
            potential defect binary map
        edge_map_ori(2D np.array, uint8): [image_height, image_width],
            original input image edge map
        imgfft_visual(2D np.array, float): [image_height, image_width],
            fourier transform visualization of input imagem
        img_rec(2D np.array, uint8): [image_height, image_width],
            the reconstucted image after filtering
        imgfft_mod_visual(2D np.array, float): [image_height, image_width],
            fourier transform visualization after fitlering
    '''
    edge_map_ori = cv2.Canny(input_img, 0.03*255, 0.3*255)
    # step 1: 2d fourier transform and visualization
    imgfft = np.fft.fftshift(np.fft.fft2(input_img))
    imgfft_abs = np.abs(imgfft)
    imgfft_visual = np.log(1 + np.abs(imgfft))

    imgfft_mod = imgfft.copy()
    height, width = input_img.shape
    # step 2: filtering frequency with high response
    if np.sum(edge_map_ori) <= 15000*255:
        logfft = imgfft_visual.reshape(-1)
        ecdf = ECDF(logfft)
        thre = 0.99
        cutoff_idx = math.floor(thre*len(ecdf.x))
        thre_logfft = ecdf.x[cutoff_idx]
        mask = [logfft > thre_logfft]
        mask = np.array(mask).reshape(height, width)
        imgfft_mod[mask] = 0
    # step 3: filtering horizontal and vertical components
    perc = 0.1
    total_abs = np.sum(np.abs(imgfft))
    # horizontal
    range_h = [np.sum(imgfft_abs[:, math.floor(height/2 + i)]) for i in range(height//2)]
    for width_h_half in range(height//2):
        width_h = width_h_half * 2
        if np.sum(range_h[1:width_h_half]) > total_abs * perc:
            break
    # vertical
    range_v = [np.sum(imgfft_abs[math.floor(height/2 + i), :]) for i in range(width//2)]
    for width_v_half in range(width//2):
        width_v = width_v_half * 2
        if np.sum(range_v[1:width_v_half]) > total_abs * perc:
            break
    imgfft_mod[:, math.floor(height/2-width_v/2):math.floor(height/2+width_v/2)] = 0 #  '|'
    imgfft_mod[math.floor(width/2-width_h/2):math.floor(width/2+width_h/2), :] = 0 # '--'
    # step 4: back to pixel domain
    img_rec = np.real(np.fft.ifft2(np.fft.ifftshift(imgfft_mod)))
    img_rec = cv2.normalize(img_rec, None, 0, 255, cv2.NORM_MINMAX)
    img_rec = np.array(img_rec).astype('uint8')
    # visualization
    imgfft_mod_visual = np.log(1 + np.abs(imgfft_mod))
    # edge detection, to find potential defect
    img_test = cv2.bilateralFilter(img_rec, 10, 50, 50)
    edge_map_defect = cv2.Canny(img_test, 0.1*255, 0.25*255)

    return edge_map_defect, edge_map_ori, imgfft_visual, img_rec, imgfft_mod_visual



#%%
# Part 1: import dataset

def get_ground_truth(data_, window_ratio=2, gt_flag=False):
    '''
    collect ground truth info:
        imageName (relative image names represented by indices),
        image nd.array,
        bbox info (extended)
    @ Args:
        data_:
            a list of dictionaries (one dictionary carries info of an image)
    '''
    # check data validity
    data_pred_gt_all = [any(dt['defects_pred_gt']) for dt in data_]
    assert any(data_pred_gt_all)
    
    img_name_list_ = [i for i in range(len(data_))]
    # read img, bbox json, layerlabel
    image_list_ = [dt['image'] for dt in data_]   # image 2d data
    bbox_list_extended_ = [extend_bbox(dt['defects_pred'],
                                       dt['defects_pred_gt'],
                                       ext_ratio=window_ratio) for dt in data_]
    if gt_flag:
        layer_label_list_ = [dt['layer'] for dt in data_]
        return img_name_list_, image_list_, bbox_list_extended_, layer_label_list_
    return img_name_list_, image_list_, bbox_list_extended_


def extend_bbox(bbox_list, bbox_gt_list, ext_ratio=1.0):
    '''
    extend bboxes by a certain ratio (ext_ratio)
    @ Args:
        bbox_list(list):
            a list of [[w0, h0], [w1, h1]],
            where point0 and point 1 refer to the upper-left and bottom-right
            corners of a bbox.
            i.e. represents all bboxes in one image.
    @ Returns:
        bbox_list_ext(list):
            a list of [[h_extended_upper_left, w_extended_upper_left],
                       [h_extended_lower_right, w_extended_lower_right]]
    '''
    bbox_list_ext = []
    for [(x10, y10), (x20, y20)], gt_flag in zip(bbox_list, bbox_gt_list):
        if gt_flag:
            weight = x20 - x10
            off_h = weight * (ext_ratio-1)/2.0
            x1 = int(max(x10 - off_h, 0))
            x2 = int(min(x20 + off_h, 479))

            height = y20 - y10
            off_w = height * (ext_ratio-1)/2.0
            y1 = int(max(y10 - off_w, 0))
            y2 = int(min(y20 + off_w, 479))
            
            bbox_list_ext.append([(y1, x1), (y2, x2)])
    return bbox_list_ext



#%%
# Part 3: get samples (images -> patches)

def get_sample(image_list_ori_, bbox_list_ori_, patch_size_, company_list_image_=[]):
    '''
    get patch samples from extended bboxes in all image
    @ Args:
        company_list_image_(list):
            a list of other acompany np.arrayassociated with image_list_ori_ and bbox_list_ori_
            Each accompany np.array's element index refers to image index,
            i.e. len(one_company_list) == len(imageList_ori_) == len(bboxList_ori_)
            e.g. [code_label_list_ori, image_name_list_ori]
    @ Returns:
        patch_list_(3D np.array): [n,h,w]
    '''
    patch_list_ = []
    patch_num = []
    for image, bbox_coord_list in zip(image_list_ori_, bbox_list_ori_): # for each img
        patch = img2patch(image, bbox_coord_list, patch_size_)
        if len(patch) == 0:
            pass
        else:
            patch_list_.extend(patch)
            patch_num.append(len(patch))

    company_list_num = len(company_list_image_)
    if len(company_list_image_) != 0:
        company_list_patch_ = [[] for i in range(company_list_num)]
        for img_idx, repeat_time in enumerate(patch_num):
            if repeat_time != 0:
                for list_idx in range(company_list_num):  # for each company list
                    company_list_patch_[list_idx].extend(\
                                       [company_list_image_[list_idx][img_idx] \
                                        for i in range(repeat_time)])
        patch_list_ = np.array(patch_list_)

        company_list_patch__ = []
        for one_list in company_list_patch_:
            one_list = np.array(one_list)
            company_list_patch__.append(one_list)
        return patch_list_, company_list_patch__
#    else:
#        patchList_ = np.array(patchList_)
    return patch_list_


def img2patch(img, bbox_coord, new_size):
    '''
    work on one single image.
    crop multiple patches for one bbox.
    Can deal with multiple bboxes in one image.
    @ Args:
        bbox_coord(list):
            A list of [[h0, w0], [h1, w1]],
            where point0 and point 1 refer to the upper-left and bottom-right
            corners of a bbox.
            i.e. represents all bboxes in one image.
    @ Returns:
        patch(3D np.array): [n,h,w]
    '''
    
#    plt.figure()
#    plt.imshow(img, cmap='gray')
#    current_axis = plt.gca()
    
    bw_def, __, __, __, __ = remove_regular_pattern_auto(img)
    patch = []
    for bbox in bbox_coord:

        # size to too small
        if (bbox[1][0]-bbox[0][0] < 32) & (bbox[1][1]-bbox[0][1] < 32):
            patch.append(cv2.resize(cropping(img, bbox), (new_size, new_size)))

        # size big enough
        else:
            bw_def_temp = wipe_surrounding(bw_def.copy(), bbox)
            height = bbox[1][0] - bbox[0][0]
            width = bbox[1][1] - bbox[0][1]

            if (height*1.0/width < 0.6) or (height*1.0/width > 1.5): # h,w varies
                patch_size = int(min(height, width) * 2)
            else:
                patch_size = int(min(height, width))
            stride = patch_size*2

            contours_wh, __ = cv2.findContours(bw_def_temp,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            contours_wh = [cont for cont in contours_wh if len(cont) > patch_size / 3]

            # contours_wh is empty
            if len(contours_wh) == 0:
                patch.append(cv2.resize(cropping(img, bbox), (new_size, new_size)))
            # contours_wh is not empty
            else:
                for cont in contours_wh:
                    cont_wh = np.squeeze(cont)
                    patch_center = [[point[1], point[0]] \
                                    for idx, point in enumerate(cont_wh) \
                                    if idx%stride == 0]
                    point_pair = [[(point[0] - patch_size//2, point[1] - patch_size//2),
                                   (point[0] + patch_size//2, point[1] + patch_size//2)] \
                                  for point in patch_center]
                    point_pair = [[regulize_point(point[0]), regulize_point(point[1])] \
                                   for point in point_pair]
                    for pair in point_pair:
                        patch.append(cv2.resize(img[pair[0][0]:pair[1][0],
                                                    pair[0][1]:pair[1][1]],
                                                (new_size, new_size)))
#                        rect = patches.Rectangle((pair[0][1], pair[0][0]), pair[1][1]-pair[0][1], pair[1][0]-pair[0][0], linewidth=1, edgecolor='r', facecolor='none')
#                        current_axis.add_patch(rect)
#        plt.show()
    return patch


def img2patch_sing_bbox(img, bbox_coord, new_size):
    '''
    work on one single image.
    crop one patch for one bbox.
    Can deal with multiple bboxes in one image.
    NOT USED
    '''
#    bw_def, __, __, __, __ = remove_regular_pattern_auto(img)
    patch = []
    for bbox in bbox_coord:
        patch.append(cv2.resize(img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]],
                                (new_size, new_size)))
    return patch



#%%
# Part 4: balance samples

def get_balanced_sample(layer_label_list_, patch_list_, company_list=[], flag='over'):
    '''
    balance samples of variant labels to be same amount
    @ Args:
        company_list(list):
            A list of other acompany np.array associated with layer_label_list_ and patch_list_
            Each accompany np.array's element index refers to image index,
            i.e. len(one_company_list) == len(layerLabelList) = len(patchList)
            e.g. [code_lable_list_ori, image_name_list_ori]
    '''
    uniq_layer_label_list = np.unique(layer_label_list_)
#    uniq_layer_label_num = len(uniq_layer_label_list)
    layer_label_list_ = np.array(layer_label_list_)
    patch_num_layerlabel = [len(np.where(layer_label_list_ == i)[0]) for i in uniq_layer_label_list]
#    print('patch_num_layerlabel', patch_num_layerlabel)

    layer_label_list_balanced_ = []
    patch_list_balanced_ = []
    company_list_num = len(company_list)
    if company_list_num != 0:
        company_list_balanced_ = [[] for i in range(company_list_num)]

    # balance samples by under-sampling
    if flag == 'under':
        patch_num = min(patch_num_layerlabel)
        sample_idx_layer = []
        for layer_idx in uniq_layer_label_list:
            sample_idx = np.where(layer_label_list_ == layer_idx)[0].tolist()
            sample_idx = np.array(random.sample(sample_idx, patch_num))
            sample_idx_layer.extend(sample_idx)
        sample_idx_layer = np.array(sample_idx_layer).reshape(-1)
        layer_label_list_balanced_ = layer_label_list_[sample_idx_layer]
        patch_list_balanced_ = patch_list_[sample_idx_layer]
        if company_list_num != 0:
            for one_list, one_list_balanced in zip(company_list, company_list_balanced_):
                one_list_balanced = one_list[sample_idx_layer]

    # balance samples by over-sampling
    else:
        patch_num = max(patch_num_layerlabel)
#        print('will balance samples of each class to number', patch_num)
        for layer_idx in uniq_layer_label_list:
            layer_label_list_balanced_.extend([layer_idx for i in range(patch_num)])
            sample_idx = np.where(layer_label_list_ == layer_idx)[0]
            sample_aug, company_list_aug = upbalancing(patch_list_,
                                                       sample_idx,
                                                       patch_num,
                                                       c_lists=company_list)
            patch_list_balanced_.extend(sample_aug)
#            print('patchList_balanced.shape', len(patchList_balanced))
            if company_list_num != 0:
                for one_list_aug, one_list_balanced in \
                zip(company_list_aug, company_list_balanced_):
                    one_list_balanced.extend(one_list_aug)

    layer_label_list_balanced_ = np.array(layer_label_list_balanced_)
    patch_list_balanced_ = np.array(patch_list_balanced_)
#    print('layer_label_list_balanced_.shape', layer_label_list_balanced_.shape)
#    print('patch_list_balanced_.shape', patch_list_balanced_.shape)

    if company_list_num != 0:
        company_list_balanced__ = []
        for one_list_balanced in company_list_balanced_:
            one_list_balanced = np.array(one_list_balanced)
            company_list_balanced__.append(one_list_balanced)
        return layer_label_list_balanced_, patch_list_balanced_, company_list_balanced__
#    else:
    return layer_label_list_balanced_, patch_list_balanced_


def upbalancing(samples, sample_idx_, total_num, c_lists=[]):
    '''
    augment samples by 4 types of implementations:
        up-down flip,
        left-right flip,
        clockwise rotation by 90 degrees,
        and anti-clockwise rotation by 90 degrees
    @ Args:
        samples (3D np.array): [n,h,w], the whole sample array
        sample_idx_ (1D np.array): [m,], indices of samples for augmentation
        total_num (int):
            the total number the selected samples will be augmented to
            (including initial selected samples)
        c_lists (list):
            A list of other acompany np.array associated with samples
            Each accompany np.array's element index refers to image index,
            i.e. len(one_company_list) == len(samples)
    '''
    samples = samples[sample_idx_]
    sample_num = len(sample_idx_)
    aug_num = total_num - sample_num

    fliplr_num = int(0.2 * aug_num)
    flipud_num = int(0.2 * aug_num)
    row_cw_num = int(0.2 * aug_num)
    rot_anti_cw_num = aug_num - fliplr_num - flipud_num - row_cw_num
#    print('sample number for each augmentatio type')
#    print('original_sample_num, fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num')
#    print(n, fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num)

    samples_augmented_ = []
    c_lists_augmented_ = [[] for i in c_lists]
    c_lists_valid = [one_list[sample_idx_] for one_list in c_lists]

    # original samples
    samples_augmented_.extend(samples)
    for one_list_valid, one_list_augmented in zip(c_lists_valid, c_lists_augmented_):
        one_list_augmented.extend(one_list_valid)

    # generate aug
    aug_type_list = ['fliplr', 'flipud', 'rot_cw', 'rot_anti_cw']
    aug_num_list = [fliplr_num, flipud_num, row_cw_num, rot_anti_cw_num]
    for one_type_aug, aug_num_sing in zip(aug_type_list, aug_num_list):
        samp, c_lists_aug = upsampling_one_type(samples, aug_num_sing, c_lists_=c_lists_valid,
                                                aug_type=one_type_aug)
        samples_augmented_.extend(samp)
#        print(len(samp))
#        print(len(c_lists_aug[0]))
        for one_list_aug, one_list_augmented in zip(c_lists_aug, c_lists_augmented_):
            one_list_augmented.extend(one_list_aug)

    samples_augmented_ = np.array(samples_augmented_)
#    print('samples_augmented_.shape', samples_augmented_.shape)
    return samples_augmented_, c_lists_augmented_


def upsampling_one_type(samples_, aug_num, c_lists_=[], aug_type=None):
    '''
    augment selected samples to a certain number (aug_num) by one augmentation implementation
    @ Args:
        samples_ (3D np.array): [m,h,w] selected sample array
    '''
    samples_collect = []
    c_lists_aug_ = [[] for one_list in c_lists_]

    sample_num = samples_.shape[0]
#    print('samples_.shape', samples_.shape)
    times = aug_num // sample_num
    if times >= 1:
        for i in range(times):
            samples_collect.extend(samples_)
            if len(c_lists_) > 0:
                for one_list_aug, one_list in zip(c_lists_aug_, c_lists_):
                    one_list_aug.extend(one_list)
    rest = aug_num % sample_num
    if rest > 0:
        idx = random.sample(range(sample_num), rest)
        samples_collect.extend(samples_[idx])
        if len(c_lists_) > 0:
            for one_list_aug, one_list in zip(c_lists_aug_, c_lists_):
                one_list_aug.extend(one_list[idx])

    samples_collect = np.array(samples_collect)
#    print(samples_collect.shape)
    if len(samples_collect) > 0:
        if aug_type == 'fliplr':
            samples_aug = np.flip(samples_collect, axis=2)
        elif aug_type == 'flipud':
            samples_aug = np.flip(samples_collect, axis=1)
        elif aug_type == 'rot_cw':
            samples_aug = np.rot90(samples_collect, k=3, axes=(1, 2))
        elif aug_type == 'rot_anti_cw':
            samples_aug = np.rot90(samples_collect, k=1, axes=(1, 2))
    else:
        samples_aug = []

    return samples_aug, c_lists_aug_



#%%
# Part 5: extract feature and do normalization  # model part2

def get_dft_feature(input_patch, feat_size):
    '''
    work for one single smaple patch
    extract dft feature for a sample patch, and resize response sample patch
    @ Args:
        input_patch(3D np.array): [n,h,w], the patches for DFT
        feat_size(int): the output patch size the initial-size DFT patch resized to
    '''
    dftfft = np.fft.fftshift(np.fft.fft2(input_patch.astype('float64')))
    return cv2.resize(np.abs(dftfft), (feat_size, feat_size))


#%%
# Part 6: train/apply classifiers  # model part3

def pred_patch2img(pred_class, image_raw_name, true_class=[]):
    '''
    summarize image-level predictions by image-wise majority vote of patch predictions
    @ Args:
        pred_class(numpy.array/list): (sampleNum,)  w.r.t. patch
        image_raw_name(numpy.array): (sampleNum,)  w.r.t. patch
    @ returns:
        pred_class_image(numpy.array): (imageNum)  w.r.t. image
        image_name_image(numpy.array): (imageNum,)  w.r.t. image
            # np.array of np.int, ascendingly sorted
    '''
    pred_class = np.array(pred_class).astype(int)  # w.r.t. patch
    true_class = np.array(true_class).astype(int)  # w.r.t. patch

    pred_class_image = []  # w.r.t. image
    true_class_image = []  # w.r.t. image
    image_name_image = np.unique(image_raw_name)  # np.array of np.int, ascendingly sorted

    for image_name in image_name_image:
        sample_idx_img = np.where(image_raw_name == image_name)[0]
        pred_class_img = pred_class[sample_idx_img]
        pred_class_image.append(np.argmax(np.bincount(pred_class_img)))
        if len(true_class) > 0:
            true_class_image.append(true_class[sample_idx_img[0]])

    pred_class_image = np.array(pred_class_image).astype(int)
    true_class_image = np.array(true_class_image).astype(int)

    if len(true_class) > 0:
        return pred_class_image, true_class_image, image_name_image
#    else:
    return pred_class_image, image_name_image
