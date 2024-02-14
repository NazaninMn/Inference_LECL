"""
Author: Nazanin Moradinasab

Inference code
"""
# import libraries
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter 
import cv2
from scipy.ndimage import measurements
from skimage import measure
from options_integration import Options
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    # set parameters
    list_dir = opt.img_name
    list_dir = list_dir.split(',')
    root = opt.root
    list_markers = opt.list_markers_name
    list_markers = list_markers.split(',')
    gaussian_filter_list = []
    for marker in list_markers:
        root_final_result = root+'/test_data/'+marker+ '/output_infer/results_final/'
        root_ = root+ '/test_data/'+marker+ '/output_infer/'
        gaussian_filter_list.append(gaussian(root_final_result,root_,list_dir))
    integrated = np.array(gaussian_filter_list).sum(0) 
    label,N = measurements.label(integrated)
    
    data_frame = pd.DataFrame(columns=['X-centre values','Y-centre values','Z-slice position','Categories']) 
    for img in list_dir:
        if img.startswith('.')!=True:

           label,N = measurements.label(integrated)
           pred_regions = measure.regionprops(label)
           pred_points = []
           for region in pred_regions:
               pred_points.append(region.centroid) 
           i = 2 # first marker 
           for marker in list_markers:
                root_final_result = root+'/test_data/'+marker+'/output_infer/results_final/'
                file_marker= pd.read_csv(root_final_result+img)
                data_frame = integrate(file_marker, label,data_frame,marker_num=i)  
                i+=1

           data_frame['X-centre values']=data_frame['X-centre values'].astype('float64')
           data_frame['Y-centre values']=data_frame['Y-centre values'].astype('float64')
           
           create_folder(root+'/test_data/integrated_results/')
           data_frame.to_csv(root+'/test_data/integrated_results/'+list_dir[0].split('.')[0]+'.csv', index=False)

        
    

def create_folder(folder):
    """ create new directory """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def gaussian(root_final_result,root, list_dir):
    
    for img in list_dir:
        if img.startswith('.')!=True:
           file = pd.read_csv(root_final_result+img)
           image = img.split('_')[1].split('.')[0]
           folder = 'results_final_'+image
           dir_img = root +  folder + '/' + image + '.png'
           y,x = np.array(Image.open(dir_img)).shape[:2] 
           mask = np.zeros((y,x)) 
           for idx in range(file.shape[0]):
               x = int(file['X-centre values'][idx]) 
               y = int(file['Y-centre values'][idx]) 
               # mask[y,x]=1 
               cv2.circle(mask, (x,y), 3, color=1, thickness=-1)  # Thickness=-1 fills the circle 
    return mask   


def integrate(file_TdTom, label,data_frame,marker_num=2):
    pred_points = []
    pred_regions = measure.regionprops(label)
    for region in pred_regions:
           pred_points.append(region.centroid) 
    for idx in range(file_TdTom.shape[0]):
           x = int(file_TdTom['X-centre values'][idx]) 
           y = int(file_TdTom['Y-centre values'][idx]) 
           labels = int(file_TdTom['Categories'][idx]) 
           mask = np.zeros(label.shape)
           mask[y,x]=1 
           index_nuclei = (label*mask).max() 
           mask_nuclei = label==index_nuclei
           pred_regions = measure.regionprops(mask_nuclei*1) 
           pred_points_one_nuclei=[] 
           for region in pred_regions:
               pred_points_one_nuclei.append(region.centroid)  
           category = 1
           if ((data_frame['X-centre values'] == round(pred_points_one_nuclei[0][1])) & (data_frame['Y-centre values'] == round(pred_points_one_nuclei[0][0]))).any():
               category = data_frame.loc[(data_frame['X-centre values'] == round(pred_points_one_nuclei[0][1])) & (data_frame['Y-centre values'] == round(pred_points_one_nuclei[0][0])), 'Categories'].item()

               if len(str(labels))==2:
                  data_frame.loc[(data_frame['X-centre values'] == round(pred_points_one_nuclei[0][1])) & (data_frame['Y-centre values'] == round(pred_points_one_nuclei[0][0])), 'Categories']=str(category)+str(marker_num) 
           else: 
               data_frame.at[idx,'X-centre values'] = round(pred_points_one_nuclei[0][1]) 
               data_frame.at[idx,'Y-centre values'] = round(pred_points_one_nuclei[0][0])  
               if len(str(labels))==2:
                   category = str(category)+str(marker_num) 
               data_frame.at[idx,'Z-slice position']=1     
               data_frame.at[idx,'Categories'] = str(category) 
    return data_frame    


if __name__ == '__main__':
    start= time.time()
    opt = Options()
    opt.parse()
    main()
    print(time.time()-start)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
