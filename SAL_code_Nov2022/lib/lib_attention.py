'''
Attention Tools
@Author: Yijing Yang
@Date: 2020.12.22
@Contact: yangyijing710@outlook.com
'''

import numpy as np
import cv2
import skimage
from skimage.morphology import disk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bbox_inline(img,bbox,ax,title='bbox',root='./',idx=0):
    # fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    # Create a Rectangle patch
    x = bbox['UL'][1]-1
    y = bbox['UL'][0]-1
    H = np.abs(bbox['UL'][0] - bbox['LL'][0])+1
    W = np.abs(bbox['LR'][1] - bbox['LL'][1])+1
    rect = patches.Rectangle((x,y), W, H, linewidth=2,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.axis('off')
    return ax

def crop_att_inline(img,bbox,target=[32,32]):
    cropped = np.copy(img[(max(bbox['UL'][0],0)):(bbox['LL'][0]+1),(max(bbox['UL'][1],0)):(bbox['UR'][1]+1)])
    cropped = cv2.resize(cropped,(target[0],target[1]),interpolation=cv2.INTER_LANCZOS4)
    return cropped

def draw_bbox_binheat(img,bbox,title='bbox',root='./',idx=0):
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(img,cmap='gray')
    # Create a Rectangle patch
    x = bbox['UL'][1]-1
    y = bbox['UL'][0]-1
    H = np.abs(bbox['UL'][0] - bbox['LL'][0])+1
    W = np.abs(bbox['LR'][1] - bbox['LL'][1])+1
    rect = patches.Rectangle((x,y), W, H, linewidth=2,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    # plt.title(title)
    plt.axis('off')
    plt.show()
    # plt.savefig(root+str(idx)+'.png')
    # plt.close()

# def draw_bbox(img,bbox,title='bbox',root='./',idx=0):
#     fig,ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(img)
#     # Create a Rectangle patch
#     x = bbox['UL'][1]
#     y = bbox['UL'][0]
#     H = np.abs(bbox['UL'][0] - bbox['LL'][0])+1
#     W = np.abs(bbox['LR'][1] - bbox['LL'][1])+1
#     rect = patches.Rectangle((x,y), W, H, linewidth=2,edgecolor='r',facecolor='none')
#     # Add the patch to the Axes
#     ax.add_patch(rect)
#     plt.axis('off')
#     plt.title(title)
#     plt.show()
#     # plt.savefig(root+str(idx)+'.png')
#     # plt.close()
#
# def draw_multi_bbox(img,bbox,root='./',title='cat',color='r'):
#     fig,ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(img)
#     # Create a Rectangle patch
#     # color = ['r','g','b','y','c','m']
#     for n in range(len(bbox)):
#         x = bbox[n]['UL'][1]
#         y = bbox[n]['UL'][0]
#         H = np.abs(bbox[n]['UL'][0] - bbox[n]['LL'][0])
#         W = np.abs(bbox[n]['LR'][1] - bbox[n]['LL'][1])
#         rect = patches.Rectangle((x,y), W, H, linewidth=2,edgecolor=color,facecolor='none')
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#     plt.title(title)
#     # plt.savefig(root)
#     # plt.close()
#     plt.show()

def get_bbox_corner(H, W, ReF, TarSize=32, Pad=0):
    STRIDE = (TarSize - (ReF-1)) // H
    corner_map = np.zeros((H, W, 4, 2))
    for i in range(H):
        for j in range(W):
            x = i * STRIDE + (ReF-1)//2
            y = j * STRIDE + (ReF-1)//2
            corner_map[i,j,0] = np.array([x-ReF//2,y-ReF//2])
            corner_map[i,j,1] = np.array([x-ReF//2,y+1+ReF//2])
            corner_map[i,j,2] = np.array([x+1+ReF//2,y-ReF//2])
            corner_map[i,j,3] = np.array([x+1+ReF//2,y+1+ReF//2])
    return corner_map


def heat2bbox_adaptive_v2(heatmap, thrs=0.6, num_bb=1):
    '''
    modify the heat2bbox criterion
    '''

    # binarization
    bin_heat = np.copy(heatmap)
    bin_heat[bin_heat < thrs] = 0
    bin_heat[bin_heat >= thrs] = 1
    # cleaning
    bin_heat = skimage.filters.median(bin_heat, disk(3))

    # ------------------- BBox regulization ------------------
    loc = np.argwhere(bin_heat > 0)
    if loc.size <25: # if too small
        loc = np.argwhere(heatmap > np.sort(heatmap.reshape(-1))[int(0.75 * 1024)])

    bbox = {}
    bbox['UL'] = [np.min(loc[:, 0]), np.min(loc[:, 1])]
    bbox['UR'] = [np.min(loc[:, 0]), np.max(loc[:, 1])]
    bbox['LL'] = [np.max(loc[:, 0]), np.min(loc[:, 1])]
    bbox['LR'] = [np.max(loc[:, 0]), np.max(loc[:, 1])]

    center_loc = [np.min(loc[:, 0])+np.max(loc[:, 0]),np.min(loc[:, 1])+np.max(loc[:, 1])]
    center_loc = np.round(np.array(center_loc)/2.0).astype('int64')
    H = np.max(loc[:, 0]) - np.min(loc[:, 0])
    W = np.max(loc[:, 1]) - np.min(loc[:, 1])

    # check the aspect ratio and finetune
    if max(H,W)<16:
        ratio = 16.0/max(H,W)
        H = int(ratio*H)
        W = int(ratio*W)

    if min(H,W)<(0.7*max(H,W)):
        bbox_H = int(max(H, (H+W)/2))
        bbox_W = int(max(W, (H+W)/2))
    else:
        bbox_H = int(H)
        bbox_W = int(W)

    minX = center_loc[0] - bbox_H//2
    maxX = center_loc[0] + bbox_H//2
    minY = center_loc[1] - bbox_W//2
    maxY = center_loc[1] + bbox_W//2
    # ----------------------------------------------------
    bbox = {}
    bbox['UL'] = [minX, minY]
    bbox['UR'] = [minX, maxY]
    bbox['LL'] = [maxX, minY]
    bbox['LR'] = [maxX, maxY]

    return bbox, bin_heat

def crop_att(img,bbox,target=[32,32]):
    cropped = np.copy(img[(max(bbox['UL'][0]-1,0)):(bbox['LL'][0]+2),(max(bbox['UL'][1]-1,0)):(bbox['UR'][1]+2)])
    cropped = cv2.resize(cropped,(target[0],target[1]),interpolation=cv2.INTER_LANCZOS4)
    return cropped



def get_prob_heatmap(probmap, ReF, TarSize=32, Pad=0):
    '''
    probmap: probability map
    ReF: receptive field size
    TarSize: target heatmap size
    Pad: padding or not # only pad=0 works now
    '''
    N, H, W = probmap.shape
    heatmaps = np.zeros((N, TarSize, TarSize))
    deno = np.zeros((1, TarSize, TarSize))
    TEMPLATE = np.ones((1, ReF, ReF))
    
    if Pad==0:
        STRIDE = (TarSize - (ReF-1)) // H
        for i in range(H):
            for j in range(W):
                x = i * STRIDE + (ReF-1)//2
                y = j * STRIDE + (ReF-1)//2
                deno[:, (x-ReF//2):(x+1+ReF//2), (y-ReF//2):(y+1+ReF//2)] += TEMPLATE
                heatmaps[:, (x-ReF//2):(x+1+ReF//2),(y-ReF//2):(y+1+ReF//2)] += probmap[:,i,j].reshape(-1,1,1) * TEMPLATE
        heatmaps = heatmaps/deno
    
    return heatmaps

def get_prob_heatmap_max(probmap, ReF, TarSize=32, Pad=0):
    '''
    probmap: probability map
    ReF: receptive field size
    TarSize: target heatmap size
    Pad: padding or not # only pad=0 works now
    '''
    N, H, W = probmap.shape
    heatmaps = np.zeros((N, TarSize, TarSize))
    deno = np.zeros((1, TarSize, TarSize))
    
    # if ReF>15:
    #     SIGMA=15#20
    # else:
    #     SIGMA=2#7
        
    TEMPLATE = np.ones((1, ReF, ReF))
    
    if Pad==0:
        STRIDE = (TarSize - (ReF-1)) // H
        for i in range(H):
            for j in range(W):
                x = i * STRIDE + (ReF-1)//2
                y = j * STRIDE + (ReF-1)//2
                deno[:, (x-ReF//2):(x+1+ReF//2), (y-ReF//2):(y+1+ReF//2)] += TEMPLATE
                # heatmaps[:, (x-ReF//2):(x+1+ReF//2),(y-ReF//2):(y+1+ReF//2)] += probmap[:,i,j].reshape(-1,1,1) * TEMPLATE
                tmp = probmap[:,i,j].reshape(-1,1,1) * TEMPLATE
                
                tmp = np.concatenate((tmp[:,:,:,np.newaxis],heatmaps[:, (x-ReF//2):(x+1+ReF//2),(y-ReF//2):(y+1+ReF//2)][:,:,:,np.newaxis]),axis=-1)
                
                heatmaps[:, (x-ReF//2):(x+1+ReF//2),(y-ReF//2):(y+1+ReF//2)] = np.max(tmp,axis=-1)
                
        mask = deno>0
        deno[deno==0]=1
        # heatmaps = heatmaps/deno
    
    if np.sum(mask==1)<(TarSize**2):
        cutH = int(np.power(np.sum(mask==1),0.5))
        heatmaps_cut = heatmaps[:,:cutH,:cutH]
        for n in range(N):
            heatmaps[n] = cv2.resize(heatmaps_cut[n], (TarSize, TarSize), interpolation=cv2.INTER_CUBIC)
    
    return heatmaps


def get_prob_heatmap_soft(probmap, ReF, TarSize=32, Pad=0):
    '''
    probmap: probability map
    ReF: receptive field size
    TarSize: target heatmap size
    Pad: padding or not # only pad=0 works now
    '''
    N, H, W = probmap.shape
    heatmaps = np.zeros((N, TarSize, TarSize))
    deno = np.zeros((1, TarSize, TarSize))
    
    if ReF>15:
        SIGMA=15#20
    else:
        SIGMA=2#7
        
    TEMPLATE = gauss2D(shape=(ReF,ReF),sigma=SIGMA).reshape((1, ReF, ReF))
    
    if Pad==0:
        STRIDE = (TarSize - (ReF-1)) // H
        for i in range(H):
            for j in range(W):
                x = i * STRIDE + (ReF-1)//2
                y = j * STRIDE + (ReF-1)//2
                deno[:, (x-ReF//2):(x+1+ReF//2), (y-ReF//2):(y+1+ReF//2)] += TEMPLATE
                heatmaps[:, (x-ReF//2):(x+1+ReF//2),(y-ReF//2):(y+1+ReF//2)] += probmap[:,i,j].reshape(-1,1,1) * TEMPLATE
        mask = deno>0
        deno[deno==0]=1
        heatmaps = heatmaps/deno
    
    if np.sum(mask==1)<(TarSize**2):
        cutH = int(np.power(np.sum(mask==1),0.5))
        heatmaps_cut = heatmaps[:,:cutH,:cutH]
        for n in range(N):
            heatmaps[n] = cv2.resize(heatmaps_cut[n], (TarSize, TarSize), interpolation=cv2.INTER_CUBIC)
    
    return heatmaps

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h













