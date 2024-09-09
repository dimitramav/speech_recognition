#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:06:37 2024

@author: dimitra
"""
from azureml.opendatasets import MsSnsd
import os

# Specify the download folder for the dataset
download_path = "./ms_snsd_sample"
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Load the dataset
ms_snsd = MsSnsd()

# Download a sample of the data (e.g., 10 files)
ms_snsd.download(download_path, max_files=10)