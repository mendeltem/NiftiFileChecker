#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:53:49 2023

@author: temuuleu
"""

import os
from pathlib import Path


# def create_masterfile(train_masterfile_path: Path,
#                       input_image: Path,
#                       mni_matrix: Path,
#                       input_mask: Path,
#                       output_dir: Path = Path(".")
#                       ) -> tuple:
#     """
#     Creates a master file with the given input parameters.

#     Args:
#         train_masterfile_path (Path): File path of the existing master file.
#         input_image (Path): Input image file path.
#         mni_matrix (Path): MNI transformation file path.
#         input_mask (Path): Input mask file path.
#         output_dir (Path): Output directory path.

#     Returns:
#         tuple: A tuple containing the new master file path, trainstring, and row number.
#     """

#     if not os.path.isdir(output_dir):
#         raise FileNotFoundError(f"The specified output directory '{output_dir}' does not exist.")


#     if not os.path.isfile(train_masterfile_path):
#         raise FileNotFoundError(f"The training master file '{train_masterfile_path}' does not exist.")

#     if not os.path.isfile(input_image):
#         raise FileNotFoundError(f"The input image file '{input_image}' does not exist.")

#     if not os.path.isfile(mni_matrix):
#         raise FileNotFoundError(f"The MNI matrix file '{mni_matrix}' does not exist.")


#     with open(train_masterfile_path, 'r') as file:
#         master_file_txt = file.read()


#     row_number = master_file_txt.lower().count("flair_to_mni.mat") + 1

#     trainstring = ",".join([str(r) for r in range(1, row_number)])

#     master_file_test = f"{input_image} {mni_matrix} {input_mask}"

#     all_master_file_txt = master_file_txt + "\n" + master_file_test + "\n"

#     new_master_file_path = os.path.join(output_dir, "master_file.txt")

#     with new_master_file_path.open('w') as f:
#         f.write(all_master_file_txt)

#     return new_master_file_path, trainstring, row_number







