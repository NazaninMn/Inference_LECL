"""
Author: Nazanin Moradinasab

Inference code
"""

import os
import numpy as np
import argparse


class Options:
    def __init__(self):
        self.outputdirectory = 'output'     # outputdirectory


    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--img_name', default=None, type=str,help='image name')
        parser.add_argument('--root', type=str, help='root directory')
        parser.add_argument('--list_markers_name', type=str, help='list markers name')
        args = parser.parse_args()

        self.root = args.root
        self.img_name = args.img_name
        self.list_markers_name = args.list_markers_name
        
