"""
Author: Nazanin Moradinasab

Inference code
"""

# import libraries
from PIL import Image
import os
import numpy as np
from skimage import io
import torch
from options import Options
import json
from read_roi import read_roi_file
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import colorsys
import subprocess
import cv2
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage import measure
import pandas as pd
import skimage.morphology as ski_morph
from misc.utils import get_bounding_box, remove_small_objects
import shutil
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    # set parameters
    patch_size = opt.patch_size
    h_overlap = opt.h_overlap
    w_overlap = opt.h_overlap
    radius = opt.radius
    outputdirectory = opt.outputdirectory
    inputdirectory = opt.inputdirectory
    img_dir = opt.root+'/'+inputdirectory+'/TIFF_wsi'
    gt_path = opt.root+'/'+inputdirectory+'/labels_point/3D'
    nuclei_channel_image = opt.nuclei_channel_image
    marker_channel_image = opt.marker_channel_image
    # marker_channel = opt.marker_channel
    bit_value = opt.bit
    root = opt.root
    marker_name = None
    test_list = [opt.test_list]
    results_final=opt.root+'/{:s}/{:s}/output_infer/results_final'.format(inputdirectory,outputdirectory)
    integrated_results = opt.root+'/{:s}/{:s}/output_infer/integrated_results/'.format(inputdirectory,outputdirectory)
    nuclei_marker_slices_rgb = opt.root+'/{:s}/{:s}/nuclei_marker_slices_rgb'.format(inputdirectory,outputdirectory)
    nuclei_marker_slices_patches = opt.root+'/{:s}/output/patches_infer'.format(inputdirectory,outputdirectory)
    rois_path = opt.root+'/'+inputdirectory+'/LesionROI/'
    tile_json_path = opt.root+'/{:s}/output/output_infer/pred/json/'.format(inputdirectory,outputdirectory)
    masks = opt.root+'/{:s}/{:s}/output_infer/masks/'.format(inputdirectory,outputdirectory)
    masks_metrics = opt.root+'/{:s}/{:s}/output_infer/mask_metric/'.format(inputdirectory,outputdirectory)
    result = opt.root+'/{:s}/{:s}/output_infer/result'.format(inputdirectory,outputdirectory)
    result_metrics = opt.root+'/{:s}/{:s}/output_infer/result_metrics/'.format(inputdirectory,outputdirectory)
    data_annotate_dir = opt.root+'/'+inputdirectory+'/SpreadsheetforCellSummary.xlsx'
    gt_point_classes = opt.root+'/{:s}/{:s}/output_infer/gt_point_classes/'.format(inputdirectory,outputdirectory)
    script_name = opt.root+'/Inference_LECL/run_tile.sh'
   
    make_executable(script_name)
    
    # prepare directories and read images and associated ROI
    if os.path.exists(nuclei_marker_slices_rgb) and os.path.isdir(nuclei_marker_slices_rgb):
        shutil.rmtree(nuclei_marker_slices_rgb)    
    create_folder(nuclei_marker_slices_rgb)
    list_slices = []
    for img_name in test_list:
        name = img_name.split('.')[0]
        marker_image = io.imread('{:s}/{:s}.tif'.format(img_dir, name))[:, marker_channel_image]
        marker_image = np.moveaxis(marker_image, 0, 2)  # move z to the end
        marker_image_roi = ROI(marker_image, name, rois_path)
        marker_image_roi = bytescale(marker_image_roi)
        nuclei_image = io.imread('{:s}/{:s}.tif'.format(img_dir, name))[:, nuclei_channel_image]
        nuclei_image = np.moveaxis(nuclei_image, 0, 2)  # move z to the end
        nuclei_image_roi = ROI(nuclei_image, name, rois_path)
        nuclei_image_roi = bytescale(nuclei_image_roi)
        for i in range(nuclei_image_roi.shape[2]):
            out = multichannel(marker_image_roi[:,:,i], nuclei_image_roi[:,:,i])
            out.save(os.path.join(nuclei_marker_slices_rgb, name) +'_slice'+str(i)+'.png')
            list_slices.append(name +'_slice'+str(i)+'.png')

    # split the images into patches
    split_patches(nuclei_marker_slices_rgb,test_list, nuclei_marker_slices_patches, patch_size, h_overlap, w_overlap, data='MO')
    
    # Run infer_code over patches
    p = subprocess.Popen(os.path.join(os.getcwd(),'run_tile.sh'), shell=True, stdin=subprocess.PIPE)
    stdout, stderr = p.communicate()
    infer(n_channels=2, list_image=list_slices, json_path = tile_json_path, tile_path=nuclei_marker_slices_patches, Masks=masks,Masks_metrics=masks_metrics)
    
    # Reconstruct images
    result=result+'_'+name
    if os.path.exists(result) and os.path.isdir(result):
        shutil.rmtree(result)
    if os.path.exists(result_metrics) and os.path.isdir(result_metrics):    
        shutil.rmtree(result_metrics)
    create_folder(result)
    create_folder(result_metrics)
    for img_name in list_slices:
        out = reconstruct(data_dir=nuclei_marker_slices_rgb, mask_dir=masks, image_name=img_name, h_overlap=h_overlap, w_overlap=w_overlap, patch_size=patch_size)
        img = Image.fromarray(out.astype('uint8'))
        img.save(os.path.join(result , img_name))
        plt.figure(figsize=(5, 5))
        plt.imshow(out)
        plt.show()
        out = reconstruct(data_dir=nuclei_marker_slices_rgb, mask_dir=masks_metrics, image_name=img_name, h_overlap=h_overlap, w_overlap=w_overlap, patch_size=patch_size, metric=True)
        print(out.shape)
        np.save(result_metrics + img_name[:-4] + '.npy', out)
        plt.figure(figsize=(5, 5))
        plt.imshow(out[:, :, 0])
        plt.show()
        
    
    unique_list, images_integrated_slices = integrate_slices(result_metrics_=result_metrics)
    nuclei_reduced_whithout_marker = reduce_slices_to_one(images_integrated_slices=images_integrated_slices, channel=0)
    marker_reduced = reduce_slices_to_one(images_integrated_slices=images_integrated_slices, channel=1)
    
    # save the results
    if os.path.exists(integrated_results) and os.path.isdir(integrated_results):
        shutil.rmtree(integrated_results)  
    create_folder(integrated_results)    
    list_img = os.listdir(img_dir) 
    
    data_frame = pd.DataFrame() 
    results_final_all = results_final
    create_folder(results_final_all)
    for indx in range(len(unique_list)):
        name = [i for i in list_img if i.startswith(unique_list[indx])][0]
        
        # nuclei
        selected_predicted_label = np.max(np.concatenate([nuclei_reduced_whithout_marker[indx][np.newaxis,:,:], marker_reduced[indx][np.newaxis,:,:]],axis=0),axis=0)
        pred_labeled_nuclei, N = measure.label(selected_predicted_label, return_num=True)
        print(N)
        np.save(integrated_results+'nuclei_'+unique_list[indx]+'.npy',np.array(pred_labeled_nuclei))
    
        # marker
        name = [i for i in list_img if i.startswith(unique_list[indx])][0]
        selected_predicted_label = marker_reduced[indx]
        pred_labeled_marker, N = measure.label(selected_predicted_label, return_num=True)
        print(N)
        np.save(integrated_results+'marker_'+unique_list[indx]+'.npy',np.array(pred_labeled_marker))
        
        #new mask
        new_mask = np.zeros_like(pred_labeled_nuclei)+(pred_labeled_nuclei>0)
        for marker in range(1,pred_labeled_marker.max()):
                index = pred_labeled_nuclei[pred_labeled_marker==marker][0]
                new_mask=new_mask+(pred_labeled_nuclei==index)
        new_mask[new_mask>2]=2
        pred_labeled = ski_morph.label(new_mask)
        pred_regions = measure.regionprops(pred_labeled)
        pred_points = []
        for region in pred_regions:
            pred_points.append(region.centroid)

        # for overlay call nuclei
        nuclei_image = io.imread('{:s}/{:s}'.format(img_dir, name))[:, nuclei_channel_image]
        nuclei_image = np.moveaxis(nuclei_image, 0, 2)  # move z to the end
        nuclei_image_roi = ROI(nuclei_image, name, rois_path)
        nuclei_image_roi = bytescale(nuclei_image_roi)
        nuclei_image_roi = nuclei_image_roi.max(-1)
        nuclei_image_roi = np.stack((nuclei_image_roi,) * 3, axis=-1)
        
        results_final=results_final+'_'+name.split('.')[0]
        if os.path.exists(results_final) and os.path.isdir(results_final):
            shutil.rmtree(results_final)  
        create_folder(results_final)
    
        for i in range(len(pred_points)):
            category=new_mask[round(pred_points[i][0]),round(pred_points[i][1])]

            data_frame.at[i,'X-centre values']=round(pred_points[i][1])
            data_frame.at[i,'Y-centre values']=round(pred_points[i][0])
            data_frame.at[i,'Z-slice position']=1
            
            if category==2:
                data_frame.at[i,'Categories']=str(12)
            elif category==1:
                data_frame.at[i,'Categories']=str(1)  
            else:   # none values are generated because of the rounding, which are modified
                data_frame.at[i,'Categories']=str(1)
            if category==1:
                nuclei_image_roi = cv2.circle(nuclei_image_roi.astype('uint8'), (
                                    round(pred_points[i][1]),round(pred_points[i][0])), 5,
                                                      (255, 255, 0), -1)
            if category==2:
                nuclei_image_roi = cv2.circle(nuclei_image_roi.astype('uint8'), (
                                    round(pred_points[i][1]),round(pred_points[i][0])), 5,
                                                      (255, 0, 0), -1)    
        nuclei_image_roi = Image.fromarray(nuclei_image_roi)
        nuclei_image_roi.save(os.path.join(results_final,name.split('.')[0]+'.png'))
        
            
        data_frame.to_csv(os.path.join(results_final,'20X_'+name.split('.')[0]+'.tif.csv'), index=False)  # Set index=False to exclude row numbers in the CSV file
        data_frame.to_csv(os.path.join(results_final_all,'20X_'+name.split('.')[0]+'.tif.csv'), index=False)
        
        if marker_name != None:
            rename_folder(root+'/{:s}/{:s}'.format(inputdirectory,outputdirectory),root+'/{:s}/{:s}'.format(inputdirectory,marker_name))
        
        
def rename_folder(old_name, new_name):
    """rename the directory"""
    try:
        os.rename(old_name, new_name)
        print(f"Folder '{old_name}' renamed to '{new_name}' successfully.")
    except FileNotFoundError:
        print(f"Folder '{old_name}' not found.")
        

def reduce_slices_to_one(images_integrated_slices, channel):
    """reduce the integrate slices into one slice using max"""
    images = []
    for i in range(len(images_integrated_slices)):
        nuclei_image = images_integrated_slices[i][:, :, :,
                       channel]  # the zero index indicates the nuclei channel and 1 is the marker
        # go over slices and perform dilation over the points
        image_dilated_slices = []
        for j in range(nuclei_image.shape[0]):
            image_dilated_slices.append(dilation(nuclei_image[j], disk(4)))
        images.append(np.max(np.asarray(image_dilated_slices), 0))
    return images

def integrate_slices(result_metrics_):
    """integrate the slices over z-axis"""
    list_slice = os.listdir(result_metrics_)
    unique_list = np.unique([img.split('_')[0] for img in list_slice])
    list_integrate_images = []
    for uni in unique_list:
        list_integrate_images.append(integrate_slices_each_image(unique_list=unique_list, list_slice=list_slice, uni=uni, result_metrics = result_metrics_))
    return unique_list , list_integrate_images


def integrate_slices_each_image(unique_list, list_slice, uni, result_metrics):
    """integrate the slices over z-axis"""
    list_slices = [i for i in list_slice if i.startswith(uni)]
    slices = []
    for slice in list_slices:
        sl = np.load(os.path.join(result_metrics, slice))   #remove small objects
        slices.append(sl)
    out = np.asarray(slices)
    return out


def reconstruct(data_dir, mask_dir, image_name, h_overlap, w_overlap, patch_size, metric = False ):
    """ integrate the  output of the model over patches to generate the WSI"""
    global wsi_image, row_image
    if not metric:
        if image_name.endswith('png'):
            image = np.array(Image.open(os.path.join(data_dir, image_name)).convert('RGB'))
            padding = int(h_overlap/2)
            a = np.concatenate([np.zeros((padding, image.shape[1], 3), dtype=np.int32), image,np.zeros((padding, image.shape[1], 3), dtype=np.int32)])
            image = np.concatenate([np.zeros((a.shape[0], padding, 3), dtype=np.int32), a,np.zeros((a.shape[0], padding, 3), dtype=np.int32)], axis=1,)
            h, w = image.shape[0], image.shape[1]
            start = int(h_overlap / 2)
            end = int(patch_size - h_overlap / 2)
            for i in range(0, h - patch_size + 200, patch_size - h_overlap):
                for j in range(0, w - patch_size + 200, patch_size - w_overlap):

                    ptch = np.array(
                        Image.open(os.path.join(mask_dir, image_name[:-4] + '_' + str(i) + '_' + str(j) + '.png')))[
                           start:end, start:end]
                    if j == 0:
                        row_image = ptch
                    else:
                        row_image = np.concatenate([row_image, ptch], axis=1)
                    if row_image.shape[1]>= w-padding :
                        if i == 0:
                            wsi_image = row_image
                        else:
                            wsi_image = np.concatenate([wsi_image, row_image], axis=0)
    else:
        if image_name.endswith('png'):
            image = np.array(Image.open(os.path.join(data_dir, image_name)).convert('RGB'))
            padding = int(h_overlap / 2)
            a = np.concatenate([np.zeros((padding, image.shape[1], 3), dtype=np.int32), image,
                                np.zeros((padding, image.shape[1], 3), dtype=np.int32)])
            image = np.concatenate([np.zeros((a.shape[0], padding, 3), dtype=np.int32), a,
                                    np.zeros((a.shape[0], padding, 3), dtype=np.int32)], axis=1, )


            h, w = image.shape[0], image.shape[1]

            start = int(h_overlap / 2)
            end = int(patch_size - h_overlap / 2)
            for i in range(0, h - patch_size + 200, patch_size - h_overlap):
                for j in range(0, w - patch_size + 200, patch_size - w_overlap):

                    ptch = np.load(os.path.join(mask_dir, image_name[:-4] + '_' + str(i) + '_' + str(j) + '.npy'))[
                           start:end, start:end]
                    if j == 0:
                        row_image = ptch
                    else:
                        row_image = np.concatenate([row_image, ptch], axis=1)

                    if row_image.shape[1]  >= w-padding:
                        if i == 0:
                            wsi_image = row_image
                        else:
                            wsi_image = np.concatenate([wsi_image, row_image], axis=0)
    return wsi_image

def infer(n_channels, list_image, json_path, tile_path,Masks, Masks_metrics):
    """ inference over the output of the model"""
    create_folder(Masks)
    create_folder(Masks_metrics)
    list_tiles = os.listdir(tile_path)
    for img in list_image:
        for name in list_tiles:
            if name.startswith(img.split('.')[0]):  # new data==>5
                if name.endswith('png'):
                    basename = name[:-4]
                    json_path_ = json_path + basename + '.json'

                    bbox_list = []
                    centroid_list = []
                    contour_list = []
                    type_list = []
                    with open(json_path_) as json_file:
                        data = json.load(json_file)
                        mag_info = data['mag']
                        nuc_info = data['nuc']
                        for inst in nuc_info:
                            inst_info = nuc_info[inst]
                            inst_centroid = inst_info['centroid']
                            centroid_list.append(inst_centroid)
                            inst_contour = inst_info['contour']
                            contour_list.append(inst_contour)
                            inst_bbox = inst_info['bbox']
                            bbox_list.append(inst_bbox)
                            inst_type = inst_info['type']
                            type_list.append(inst_type)
                    image = np.array(Image.open(tile_path + '/' + basename + '.png'))

                    # if there is no instance in the image
                    if len(centroid_list) == 0:
                        # overlay on the RGB image
                        overlay_wsi = np.zeros((image.shape[0], image.shape[1], 3))
                        overlay_wsi = overlay_wsi.astype('uint8')
                        overlay_wsi = Image.fromarray(overlay_wsi)
                        overlay_wsi.save(Masks + name)

                        # metric calculations
                        overlay_metric = np.zeros((image.shape[0], image.shape[1], n_channels))
                        np.save(Masks_metrics + name[:-4]+ '.npy', overlay_metric)

                        continue

                    # if there is at least one instance in the image
                    overlay_wsi = image
                    overlay_metric = np.zeros((image.shape[0], image.shape[1], n_channels))

                    for i in range(len(centroid_list)):
                        rand_centroid = centroid_list[i]
                        rand_contour = contour_list[i]
                        rand_type = type_list[i]

                        # draw the overlays
                        overlay_metric[np.round(rand_centroid[1]).astype('int'), np.round(rand_centroid[0]).astype('int'), rand_type-1] = 255
                        if rand_type == 1:
                            overlay_wsi = cv2.drawContours(overlay_wsi.astype('uint8'), [np.array(rand_contour)], -1,
                                                           (255, 255, 255),
                                                           1)
                            overlay_wsi = cv2.circle(overlay_wsi.astype('uint8'), (
                            np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3,
                                              (255, 255, 255), -1)
                        elif rand_type == 2:
                            overlay_wsi = cv2.drawContours(overlay_wsi.astype('uint8'), [np.array(rand_contour)], -1,
                                                           (255, 0, 0),
                                                           1)
                            overlay_wsi = cv2.circle(overlay_wsi.astype('uint8'), (
                            np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3,
                                              (0, 255, 0), -1)
                        elif rand_type == 3:
                            overlay_wsi = cv2.drawContours(overlay_wsi.astype('uint8'), [np.array(rand_contour)], -1,
                                                           (255, 0, 255),
                                                           1)
                            overlay_wsi = cv2.circle(overlay_wsi.astype('uint8'), (
                            np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3,
                                              (255, 0, 255), -1)
                        elif rand_type == 4:
                            overlay_wsi = cv2.drawContours(overlay_wsi.astype('uint8'), [np.array(rand_contour)], -1,
                                                           (255, 255, 255),
                                                           1)
                            overlay_wsi = cv2.circle(overlay_wsi.astype('uint8'), (
                            np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3,
                                              (255, 255, 255), -1)

                    overlay_wsi = Image.fromarray(overlay_wsi)
                    overlay_wsi.save(Masks + name)
                    np.save(Masks_metrics + name[:-4]+ '.npy', overlay_metric)

def multichannel(marker, nuclei):
        """ convert multichannel images into RGB images """
        channel1 = marker
        channel2 = np.zeros((nuclei.shape[0], nuclei.shape[1]))
        channel3 = nuclei
        channel4 = np.zeros((nuclei.shape[0], nuclei.shape[1]))
        img = np.concatenate(
            [channel1[np.newaxis, :], channel2[np.newaxis, :], channel3[np.newaxis, :], channel4[np.newaxis, :]])
        img = np.moveaxis(img, 0, 2)
        n_channels = img.shape[2]
        colors = np.array(generate_colors(n_channels))
        out_shape = list(img.shape)
        out_shape[2] = 3  ## change to RGB number of channels (3)
        out = np.zeros(out_shape)
        for chan in range(img.shape[2]):
            out = out + np.expand_dims(img[:, :, chan], axis=2) * np.expand_dims(colors[chan] / 255, axis=0)
        out = out / np.max(out)
        out = Image.fromarray((out * 255).astype(np.uint8))
        return out


def generate_colors(class_names):
    """ generate colors"""
    hsv_tuples = [(x / class_names, 1., 1.) for x in range(class_names)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def ROI(ori_image, img_name, rois_path):
    """ read the ROI file """
    name = img_name.split('.')[0]
    img_size0, img_size1 = ori_image.shape[0], ori_image.shape[1]
    mask_size = np.zeros([img_size0, img_size1])
    roi = read_roi_file(rois_path + name + '.roi')
    x = np.array(roi[name]['x'])
    y = np.array(roi[name]['y'])
    n = x.shape[0]
    poly = np.zeros([n, 2], dtype=int)
    poly[:, 0] = x
    poly[:, 1] = y
    mask = Image.new('L', (img_size1, img_size0), 0)  # size = (width, height)
    polygontoadd = poly
    polygontoadd = np.reshape(polygontoadd, np.shape(polygontoadd)[0] * 2)  # convert to [x1,y1,x2,y2,..]
    polygontoadd = polygontoadd.tolist()
    ImageDraw.Draw(mask).polygon(polygontoadd, outline=1, fill=1)
    mask = np.array(mask)
    mask_size += mask
    reagion_interest = mask_size[:, :, np.newaxis] * ori_image
    return reagion_interest


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    convert images pixel values into 0-255
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, default=None
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, default=None
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, default=None
        Scale max value to `high`.  Default is 255.
    low : scalar, default=None
        Scale min value to `low`.  Default is 0.
    Returns

    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def split_patches(data_dir, test_list, save_dir, patch_size, h_overlap, w_overlap, data='MO'):
    """ split large image into small patches """
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    create_folder(save_dir)
    image_list = os.listdir(data_dir)
    # split the image
    for img in test_list:
        for image_name in image_list:
            if img[:5] in image_name:
                if image_name.endswith('png'):
                    print(image_name)
                    name = image_name.split('.')[0]
                    image_path = os.path.join(data_dir, image_name)
                    image = np.array(Image.open(image_path).convert('RGB'))
                    padding = int(h_overlap/2)
                    a = np.concatenate([np.zeros((padding, image.shape[1], 3), dtype=np.int32), image,   
                                            np.zeros((padding, image.shape[1], 3), dtype=np.int32)])
                    image = np.concatenate(
                            [np.zeros((a.shape[0], padding, 3), dtype=np.int32), a, np.zeros((a.shape[0], padding, 3), dtype=np.int32)],
                            axis=1)
                    seg_imgs = []
                    h, w = image.shape[0], image.shape[1]
                    for i in range(0, h + 1, patch_size - h_overlap):
                        for j in range(0, w + 1, patch_size - w_overlap):
                            if len(image.shape) >= 3:
                                patch = image[i:i + patch_size, j:j + patch_size, :]
                            else:
                                patch = image[i:i + patch_size, j:j + patch_size]
                            seg_imgs.append(patch)
                            io.imsave('{:s}/{:s}_{:d}_{:d}.png'.format(save_dir, name, i, j), patch)
                elif image_name.endswith('npy'):
                    name = image_name.split('.')[0]
                    image_path = os.path.join(data_dir, image_name)
                    image = np.load(image_path)
                    print(image.shape)

                    padding = int(h_overlap / 2)
                    a = np.concatenate([np.zeros((padding, image.shape[1], image.shape[2]), dtype=np.int32), image,
                                            np.zeros((padding, image.shape[1], image.shape[2]), dtype=np.int32)])
                    image = np.concatenate([np.zeros((a.shape[0], padding, a.shape[2]), dtype=np.int32), a,
                                                np.zeros((a.shape[0], padding, a.shape[2]), dtype=np.int32)], axis=1)
                    seg_imgs = []
                    h, w = image.shape[0], image.shape[1]
                    for i in range(0, h + 1, patch_size - h_overlap):
                        for j in range(0, w + 1, patch_size - w_overlap):
                            if len(image.shape) >= 3:
                                patch = image[i:i + patch_size, j:j + patch_size, :]
                            else:
                                patch = image[i:i + patch_size, j:j + patch_size]
                            seg_imgs.append(patch)
                            np.save('{:s}/{:s}_{:d}_{:d}.npy'.format(save_dir, name, i, j), patch)


def make_executable(script_name):
    try:
        subprocess.run(["chmod", "u+x", script_name])
        print(f"Changed permissions of '{script_name}' successfully.")
    except FileNotFoundError:
        print(f"Error: '{script_name}' not found.")
        
if __name__ == '__main__':
    start= time.time()
    opt = Options()
    opt.parse()
    main()
    print(time.time()-start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
