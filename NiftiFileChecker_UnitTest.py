#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:03:53 2023

@author: temuuleu
"""

import matplotlib.pyplot as plt
from nipype.interfaces import cat12
import unittest
import tempfile
import os
from NiftiFileChecker import NiftiFileChecker
import numpy as np
import shutil
#all test data directory
import nibabel as nib
import numpy as np
import subprocess
from config.config import Config


config = Config()
dir(config)
config.TEST_DATA_DIR


class TestCheckPathExists(unittest.TestCase):
    

    def setUp(self):
        TEST_DATA_DIR    = config.TEST_DATA_DIR
        ORIGINAL_DIR_NAME = "original_dir"
        TEST_DIR_NAME = "test_dir"
        
        # Define the paths for the original and test data
        self.original_dir = os.path.join(TEST_DATA_DIR, ORIGINAL_DIR_NAME)
        self.test_dir = os.path.join(TEST_DATA_DIR, TEST_DIR_NAME)

        # If test_dir exists, remove it to ensure it's not present before running the tests
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # Define the paths for the original images
        self.original_flair_image_path = os.path.join(self.original_dir, "FLAIR.nii.gz")
        self.original_flair_roi_image_path = os.path.join(self.original_dir, "FLAIR_ROI_0px.nii.gz")
        self.original_mprage_image_path = os.path.join(self.original_dir, "MPRAGE.nii")

        # Define the paths for the test images
        self.test_flair_image_path = os.path.join(self.test_dir, "sub_FLAIR.nii.gz")
        self.test_mprage_image_path = os.path.join(self.test_dir, "sub_MPRAGE.nii")
        # Define the paths for the files used in tests
        
        self.empty_nifti_path = os.path.join(self.original_dir, "empty.nii.gz")  # Assuming there is an empty file here
        self.wrong_nifti_path = os.path.join(self.original_dir, "wrong.nii.gz")  # Assuming there is a wrong file here
        self.txt_file_path = os.path.join(self.original_dir, "text.txt")  # Assuming there is a txt file here
        
        empty_image = nib.Nifti1Image(np.empty((0,0,0)), np.eye(4)) # Creating an empty NIfTI image
        nib.save(empty_image, self.empty_nifti_path)
        

        with open(self.txt_file_path, 'w') as f:
            f.write('This is a text file.')
            
        subprocess.run(["touch", self.wrong_nifti_path])


    def test_create_checker_with_wrong_datatype(self):
        # Test that the NiftiFileChecker raises an exception when given a text file
        with self.assertRaises(ValueError):  # Replace SomeException with the exception you're expecting
            NiftiFileChecker(self.txt_file_path)

    def test_if_nifti_is_empty(self):
        # Test that the NiftiFileChecker raises an exception when given an empty NIfTI file
        with self.assertRaises(ValueError):  # Replace AnotherException with the exception you're expecting
            NiftiFileChecker(self.empty_nifti_path)


    def test_if_wrong_niftii_file(self):
        with self.assertRaises(nib.filebasedimages.ImageFileError):
            NiftiFileChecker(self.wrong_nifti_path)      
            
        
    def test_image_copy_if_dir_not_exists(self):
        # Test that the image was copied correctly
        original_flair_image = NiftiFileChecker(self.original_flair_image_path)
        
        with self.assertRaises(FileNotFoundError):
            original_flair_image.copy(self.test_dir)
           
            
    def test_image_copy_if_dir_not_exists_but_force_it(self):
        # Test that the image was copied correctly
        original_flair_image = NiftiFileChecker(self.original_flair_image_path)
        
        with self.assertRaises(FileNotFoundError):
            original_flair_image.copy(self.test_dir, create_dir=True)    
            
            
            
    def test_image_copy_if_dir_exists(self):
        # Test that the image was copied correctly
        original_flair_image = NiftiFileChecker(self.original_flair_image_path)

        with self.assertRaises(FileNotFoundError):
            original_flair_image.copy(self.test_dir)       
          
            
    def test_flirt_with_original_locked(self):
        # try to flirt if it is locked
        original_flair_image = NiftiFileChecker(self.original_flair_image_path)
        
        with self.assertRaises(ValueError):
            original_flair_image.flirt()


    def test_bet_with_original_locked(self):
        # try to flirt if it is locked
        original_flair_image = NiftiFileChecker(self.original_flair_image_path)
        
        with self.assertRaises(ValueError):
            original_flair_image.bet()     



          
if __name__ == '__main__':
    unittest.main()
        
            
    # def test_add_mask(self):
    #     # Add a mask to the test image
    #     self.test_flair_image.add_mask(self.original_flair_roi_image_path)
        
    #     # Check that the mask has been added correctly
    #     self.assertTrue(self.test_flair_image._hasmask)

    # def test_flirt(self):
    #     # Apply flirt to the test image and verify the operation
    #     self.test_flair_mni_image = self.test_flair_image.flirt(force=True)
        
    #     # Assume flirt() changes the _hasmask attribute to False. If not, modify accordingly
    #     self.assertFalse(self.test_flair_mni_image._hasmask)

    # def test_nulling(self):
    #     # Apply nulling to the test image and verify the operation
    #     self.test_flair_mni_image_null = self.test_flair_mni_image.nulling()
        
    #     # Assume nulling() changes the _hasmask attribute to True. If not, modify accordingly
    #     self.assertTrue(self.test_flair_mni_image_null._hasmask)

    # def test_bet(self):
    #     # Apply bet to the test image and verify the operation
    #     self.test_flair_brain_null_image = self.test_flair_mni_image_null.bet(force=True)

    #     # Assume bet() changes the _hasmask attribute to False. If not, modify accordingly
    #     self.assertFalse(self.test_flair_brain_null_image._hasmask)


        
    # # Teardown
    # def tearDown(self):
    #     shutil.rmtree(self.test_file_files_dir)  # clean up by removing the test directory


        


            

