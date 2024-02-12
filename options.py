"""
Author: Nazanin Moradinasab

Inference code
"""
import os
import numpy as np
import argparse


class Options:
    def __init__(self):
        # self.outputdirectory = 'lineage'     # outputdirectory
        self.inputdirectory = 'test_data' 
        self.radius = 15
        self.patch_size = 256
        self.h_overlap = 20
        self.train = dict()


    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--marker_name', default=None, type=str,help='marker name')
        parser.add_argument('--nuclei_channel_image', type=int, help='nuclei channel image')
        parser.add_argument('--marker_channel_image', type=int, help='marker channel image')
        parser.add_argument('--bit', default= '16bit', type=str, help='16bit')
        parser.add_argument('--root', type=str, help='root directory')
        parser.add_argument('--test_list', type=str, help='image name')
        # parser.add_argument('--outputdirectory', type=str, help='image name')
        args = parser.parse_args()

        self.bit = args.bit
        self.root = args.root
        # self.marker_name = args.marker_name
        self.outputdirectory = args.marker_name
        self.test_list = args.test_list
        self.nuclei_channel_image = args.nuclei_channel_image-1
        self.marker_channel_image= args.marker_channel_image-1
        
