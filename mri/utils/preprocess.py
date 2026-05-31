import sigpy as sp
import os
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
from tqdm import tqdm
import json
import sys
import os
import sys
import subprocess

BART_PATH1 = "/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart/bart"
BART_PATH2 = "/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart/"
# Ensure the binary is executable
os.chmod(BART_PATH1, 0o755)
os.chmod(BART_PATH2, 0o755)


# Set necessary paths
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TOOLBOX_PATH'] = '/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart'
sys.path.append('/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart/python')

# Check if BART is accessible
subprocess.run(["/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart/bart", "version"], check=True)

# Import BART
from bart import bart
 
from multiprocessing import Pool
import random
import zipfile