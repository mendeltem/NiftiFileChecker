#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:29:01 2023

@author: temuuleu
"""

import importlib

def check_dependencies(dependencies):
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
            print(f"Successfully imported {dependency}")
        except ImportError:
            print(f"Dependency {dependency} is missing. Please install with pip:")
            print(f"pip install {dependency}")
            
            
# Example usage
dependencies = ["numpy", "matplotlib", "sklearn","nipype","nibabel"]
check_dependencies(dependencies)     

import subprocess

def check_fsl():
    try:
        # 'flirt -version' is a command that should work if FSL is installed,
        # and it returns the version of FSL.
        result = subprocess.run(['flirt', '-version'], stdout=subprocess.PIPE)
        print(f"FSL version: {result.stdout.decode('utf-8')}")
        
        
        logging.info(f"FSL version: {result.stdout.decode('utf-8')}")
        
        return 1
    except:
        print("FSL is not installed or not found in system PATH")
        return 0




import os
import shutil
import tempfile
import numpy as np
import logging


import nibabel as nib
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype import Node, Workflow
from nipype.interfaces.fsl import BET
from config.config import Config
import matplotlib.pyplot as plt
import ants
import time
from optparse import OptionParser
import inspect
from typing import Optional, Tuple

parser = OptionParser()

# Configuring logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Check FSL
fsl_installed = check_fsl()


config = Config()
dir(config)
config.masterfile_file_path


def check_path_exists(file_path , endswith = [".nii",".nii.gz"]):

    path_components = [d for d in file_path.split('/') if not "." in d ]
    
    basename = os.path.basename(file_path)
    
    if "." in basename:
        filetype = "file"
    else:
        filetype = "dir"
    
    if filetype == "file":
        #Check if file exists
        if not os.path.isfile(file_path) and not file_path.endswith(endswith[0] )  or not file_path.endswith(endswith[1] ):
             raise FileNotFoundError(f"File does not exist: {file_path}")
    
    else:
        # Check each subdirectory
        for i in range(0, len(path_components)):
            sub_path = '/'.join(path_components[:i])
            #print(f"{sub_path} {os.path.isdir(sub_path)}  {len(sub_path) }" )
            
            if not os.path.isdir(sub_path) and  not len(sub_path) == 0:
                raise FileNotFoundError(f"Subdirectory does not exist: {sub_path}")
        

class NiftiFileChecker():
    """
    Class for checking and processing NIfTI image files.
    """
        
    def __init__(self,  path_image: str,
                 image_type: Optional[str] = None, 
                 locked: bool = True ,
                 mni_mat: str =None ,
                 mask_path: str = None,
                 max_displayed_slices: int=100):
        
        
        """
        Initialize the NiftiFileChecker object with a file path and other properties.
        Checks if the the nifti exists,  niftii is not corrupt, 
        niftii not empty
        
        If the instance is create the data is locked to protect 
        original image. Please create a copyy to another directory in order to work
        with this file.
        
        
        """
        self._path = None
        self._image_array = None
        self._shape = ()
        self.__affine = None
        self._basename = os.path.basename(path_image)
        self._filename = os.path.basename(path_image).split(".")[0]
        self.standard_space_flair  = config.standard_space_flair
        
        self._dir = os.path.dirname(path_image)
        self._locked = locked
        self._mni_mat = mni_mat
  
        if not self._basename.endswith((".nii", ".nii.gz")):
            raise ValueError("Invalid file type. Only .nii and .nii.gz files are supported.")
        elif self._basename.endswith(".nii.gz"):
            self._endswith = ".nii.gz"
        elif self._basename.endswith(".nii"):
            self._endswith = ".nii"
        
        check_path_exists(path_image)
        self._check_nifti_file_image(path_image)
        self._check_nifti_file_empty()
        
        if image_type:
            self._image_type = image_type
        else:
            self._set_image_type()
            
            
        self.num_slices = self._image_array.shape[-1]
        self._voxel_count, self._volume_mm3, self._volume_ml =  self._get_volume()
        self._voxel_count_slices, self._volume_mm3_slices, self._volume_ml_slices = self._get_volume_slices(2)
    
        self._hasmask = None

        if mask_path:
            self._check_nifti_file_mask(mask_path)

        self.max_displayed_slices = max_displayed_slices
        
        
    def __str__(self):
        return str(self.path)
    
    def __repr__(self):
        return str(self.path)
       
        
    def _check_nifti_file_image(self, file_path: str) -> bool:
        """
        Checks if the file at the given file path is a valid NIfTI file.
        """
        
        nii = nib.load(file_path)
        if isinstance(nii, nib.Nifti1Image):
            print(f"File at {file_path} is valid.")
            self._path = file_path
            self._nii = nii
            self._image_array = nii.get_fdata()
            self._shape = self._image_array.shape
            self.__affine = nii.affine
        else:
            logging.error(f"File at {file_path} is not a valid NIfTI image.")
            raise ValueError(f"File at {file_path} is not a valid NIfTI image.")

    
    def _check_nifti_file_mask(self, mask_path: str) -> bool:
        """
        Checks if the file at the given file path is a valid NIfTI file.
        """
        
        nii = nib.load(mask_path)
        if isinstance(nii, nib.Nifti1Image):
            print(f"File at {mask_path} is valid.")
            
            mask_array =  nii.get_fdata()
            
            if not mask_array.shape == self._image_array.shape:
                logging.error("New image and mask not in same dimension")
                ValueError("New image and mask not in same dimension")
            
            self._mask_path = mask_path
            self._mask_basename = os.path.basename(mask_path)
            self._mask_filename =  self._mask_basename.split(".")[0]
            
            self._mask_nii = nii
            self._mask_array = nii.get_fdata()
            self._mask_shape = self._image_array.shape
            self.__mask_affine = nii.affine
            self._hasmask = True
        else:
            logging.error(f"File at {mask_path} is not a valid NIfTI image.")
            raise ValueError(f"File at {mask_path} is not a valid NIfTI image.")



    def _check_nifti_file_empty(self) ->bool:
        """
        Checks if the NIfTI file is empty.
        """
        if np.sum(self._image_array) == 0:
            logging.error("Input image is all zeros.")
            raise ValueError("Input image is all zeros.")
            
            
    def split(self, sep=None):
        return self._path.split(sep)
    
    def capitalize(self):
        return self._path.capitalize()
    
    def replace(self, old, new, count=-1):
        return self._path.replace(old, new, count)


    @property
    def strip(self, chars=None):
        return self._path.strip(chars)
    
    @property
    def lower(self):
       return self._path.lower()
   
    @property
    def upper(self):
       return self._path.upper()   

    @property
    def nii(self):
        return self._nii
    
    @property
    def image_type(self):
        return self._image_type
    

    def volume_mm3(self, slice_number=None):
        if slice_number:
           return self._volume_mm3_slices[slice_number]
        return self._volume_mm3
    

    def voxel_count(self, slice_number=None):
        if slice_number:
           return self._voxel_count_slices[slice_number]
        return self._voxel_count


    def volume_ml(self, slice_number=None):
        
        if slice_number:
            return self._volume_ml_slices[slice_number]
        return self._volume_ml


    def _set_image_type(self) -> None:
        """
        Sets the image type of the NIfTI file based on its file name.
        """
        basename_lower = self._basename.lower()
        
        if "roi" in basename_lower:
            self._image_type = "ROI"
        elif "flair" in basename_lower:
            self._image_type = "FLAIR"
        elif "dwi" in basename_lower:
            self._image_type = "DWI"
        elif "mprage" in basename_lower:
            self._image_type = "MGRAGE"
        else:
            self._image_type = "unknown"

    @property
    def path(self):
        return self._path
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def basename(self):
        return self._basename

    @property
    def affine(self):
        return self.__affine

    @property
    def get_dir(self):
        return self._dir
    
    @property
    def endswith(self):
        return self._endswith
    
    @property
    def filename(self):
        return self._filename

    def save(self, new_path_image: str) -> None:
        nib.save(self._nii1, new_path_image)
        
            
    def copy(self, new_path_image: str, create_dir: bool = False) -> 'NiftiFileChecker':
        """
        Copies the NIfTI file to a new destination.
    
        Parameters:
        - new_path_image (str): New path and filename for the copied NIfTI file.
        - create_dir (bool): Whether to create the destination directory if it doesn't exist. Default is False.
    
        Raises:
        - ValueError: If the destination directory does not exist or if the new path is the same as the current path.
    
        Returns:
        - NiftiFileChecker object representing the copied NIfTI file that is not locked.
        """
 
        basename = os.path.basename(new_path_image)
        
        if "." in basename:
            filetype = "file"
        else:
            filetype = "dir"
        
        
        if filetype == "dir":
            new_path_image = os.path.join(new_path_image, os.path.basename(self._path))
            
        print(f"new_path_image : {new_path_image}")
    
        if create_dir:
            os.makedirs(os.path.dirname(new_path_image), exist_ok=True)
            
        check_path_exists(new_path_image)
    
        if new_path_image == self.path:
            logging.error("Please provide a new path that is different from the current path.")
            raise ValueError("Please provide a new path that is different from the current path.")
    
        copy_cmd = [
            "cp",
            self._path,
            new_path_image,
        ]
    
        try:
            subprocess.run(copy_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Error copying file:") from e
    
        check_path_exists(new_path_image)
    
        return NiftiFileChecker(new_path_image, locked=False)
    

    def _get_volume(self, rounded: int =10) -> Tuple[int, float]:
        output_all_matter = subprocess.run(["fslstats", self._path, "-V"], capture_output=True)
        result_str = output_all_matter.stdout.decode("utf-8")
        all_matter_result = result_str.split()

        voxel_count = float(all_matter_result[0])
        volume_mm3 = float(all_matter_result[1]) 
        volume_ml = volume_mm3 / 1000  # convert from mm^3 to mL
        
        return   voxel_count,volume_mm3,volume_ml
    

    def _get_volume_slices(self, rounded: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the volume for each slice in the NIfTI file.
    
        Parameters:
        - rounded (int): Number of decimal places to round the volume values. Default is 10.
    
        Returns:
        - Tuple containing the voxel count for each slice and the volume in milliliters (ml) for each slice.
        """
        nifti_image = nib.load(self._path)
        volume_data = nifti_image.get_fdata()
    
        voxel_size = np.prod(nifti_image.header.get_zooms())
        voxel_count = np.sum(volume_data != 0, axis=(0, 1))  # Voxel count for each slice
        volume_mm3 = voxel_count * voxel_size  # Volume in cubic millimeters (mm^3) for each slice
        volume_ml = volume_mm3 / 1000  # Volume in milliliters (ml) for each slice
    
        return np.round(voxel_count, rounded), np.round(volume_mm3, rounded), np.round(volume_ml, rounded)    
        
    
    def bias_correction(self) -> None:
        """
        Performs N4 bias field correction on the NIfTI file using the ANTs N4BiasFieldCorrection tool.
        
        """
        
        if self._locked:
            logging.error("The data and file are locked. Please copy them somewhere.")
            raise ValueError("The data and file are locked. Please copy them somewhere.")
            
        if self._image_type ==  "ROI":
            raise ValueError("Data is ROI and cannot be used for Brain extraction")
        try:
            biascorr = Node(N4BiasFieldCorrection(save_bias=False, num_threads=-1), name="biascorr")
            biascorr.inputs.input_image = self._path
            biascorr.inputs.output_image = self._path
            biascorr.run()
        except Exception as e:
            RuntimeError("Error during N4BiasFieldCorrection:", str(e))


    def bet(self, frac: float =0.5, force: bool =False) -> None:

        """
        Performs brain extraction on the NIfTI file using the FSL BET tool.
        
        Parameters:
        - frac (float): Fractional intensity threshold. Default is 0.5.
        - force (bool): Whether to force the brain extraction even if the output file already exists. Default is False.
        
        Raises:
        - ValueError: If the data and file are locked or if the data is an ROI and cannot be used for brain extraction.
        
        Returns:
        - NiftiFileChecker object representing the extracted brain image.
        """
        
        if self._locked:
            logging.error("The data and file are locked. Please copy them somewhere.")
            raise ValueError("The data and file are locked. Please copy them somewhere.")
            
        # if self._image_type ==  "ROI":
        #     raise ValueError("Data is ROI and cannot be used for Brain extraction")
            
        print(f"Input Bet: {self._path}  {os.path.isfile(self._path)}")
        
        output_file_path = os.path.join(self._dir, self._filename + "_brain.nii.gz")
        print(f"Output Bet: {output_file_path}  {os.path.isfile(output_file_path)}")

        if not os.path.isfile(output_file_path) or force:
            if os.path.isfile(output_file_path):
                print("Forcing brain extraction. Removing existing file.")
                subprocess.run(["rm", output_file_path])
            
            try:
                # Skullstrip
                skullstrip = Node(BET(mask=False), name="skullstrip")
                skullstrip.inputs.frac = frac
                skullstrip.inputs.in_file = self._path
                skullstrip.inputs.out_file = output_file_path
                skullstrip.run()
                
                self._endswith  = ".nii.gz"
            except Exception as e:
                logging.error(f"Error during skullstrip: {str(e)}")
                RuntimeError("Error during skullstrip:", str(e))

        else:
            print("Brain already extracted.")
        
        return NiftiFileChecker(output_file_path, locked=False)
    
    
    def flirt(self, reference_image:str = "", force: bool=False):
        """
        Performs linear registration on the NIfTI file using the FSL FLIRT tool.
        
        Parameters:
        - reference_image (str): Path to the reference image for registration. If not provided, the standard space FLAIR image will be used. Default is an empty string.
        - force (bool): Whether to force the registration even if the output files already exist. Default is False.
        
        Raises:
        - ValueError: If the data and file are locked.
        
        Returns:
        - Tuple of NiftiFileChecker object representing the registered image and the path to the transformation matrix file.
        """
         
        if self._locked:
            logging.error("The data and file are locked. Please copy them somewhere.")
            raise ValueError("The data and file are locked. Please copy them somewhere.")
    
        MNI_xfm_path = os.path.join(self._dir, self._filename + "_to_registered.mat")
        mni_image_path = os.path.join(self._dir, self._filename + "_registered" + self._endswith)
        
        mni_mask_path = os.path.join(self._dir, self._mask_filename + "_registered" + self._endswith)
    
        if not reference_image:
            reference_image = self.standard_space_flair
            MNI_xfm_path = os.path.join(self._dir, self._filename + "_to_mni.mat")
            mni_image_path = os.path.join(self._dir, self._filename + "_mni" + self._endswith)
    
        if not (os.path.isfile(MNI_xfm_path)) or not (os.path.isfile(mni_image_path)) or force:
            flirt_cmd = [
                "flirt",
                "-ref", reference_image,
                "-in", self._path,
                "-omat", MNI_xfm_path,
                "-out", mni_image_path
            ]
            
            try:
                subprocess.run(flirt_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                
                logging.error(f"Error during flirt normalization:  {e}")
                RuntimeError("Error during flirt normalization:", e)
            
            if self._hasmask:
                flirt_mask_cmd = [
                    "flirt",
                    "-ref", mni_image_path,
                    "-in", self._mask_path,
                    "-out", mni_mask_path,
                    "-init",MNI_xfm_path,
                    "-applyxfm"
                ]
                try:
                    subprocess.run(flirt_mask_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error during flirt normalization:  {e}")
                    print("Error during flirt mask normalization:", e)
                    
                    RuntimeError("Error during flirt mask normalization:", str(e))

                    
                return NiftiFileChecker(mni_image_path, 
                                        locked=False, 
                                        mni_mat= MNI_xfm_path,
                                        mask_path = mni_mask_path,
                                        )     
        else:
            print("flirt outputs already exist")


        return NiftiFileChecker(mni_image_path, locked=False, mni_mat= MNI_xfm_path)
            
    
    
    def destroy(self):
        """
         Deletes the NIfTI file.
         """
        
        if not self._locked:
            print(f"deleting the file {self._path}")
            os.remove(self._path)

   
    def convert_nii_gz_to_nii(self):
        """
        Converts the NIfTI-GZ file to NIfTI format (.nii).
        
        Raises:
        - ValueError: If the data and file are locked.
        
        Returns:
        - NiftiFileChecker object representing the converted NIfTI file.
        """
        if self._locked:
            logging.error(f"The data and file are locked. Please copy them somewhere.")
            raise ValueError("Data and file locked.")
            
        nii_path = os.path.join(self._dir, self._filename + ".nii")
        nib.save(self._nii, nii_path)
        return NiftiFileChecker(nii_path, locked=False) 
    
    
    def convert_nii_to_nii_gz(self):
        
        """
        Converts the NIfTI file to NIfTI-GZ format (.nii.gz).
        
        Raises:
        - ValueError: If the data and file are locked.
        
        Returns:
        - NiftiFileChecker object representing the converted NIfTI-GZ file.
        """
        
        if self._locked:
            logging.error("The data and file are locked. Please copy them somewhere.")
            raise ValueError("Data and file locked.")
            
        nii_path = os.path.join(self._dir, self._filename + ".nii.gz")
        nib.save(self._nii, nii_path)
        return NiftiFileChecker(nii_path, locked=False) 


    def fast(self):
        """
        Performs tissue segmentation on the NIfTI file using the FSL FAST tool.
        
        Raises:
        - ValueError: If the data and file are locked or if the data is an ROI and cannot be used for brain extraction.
        
        Returns:
        - Tuple of NiftiFileChecker objects representing the segmented CSF, WM, and GM files.
        """
        if self._locked:
            logging.error("The data and file are locked. Please copy them somewhere.")
            raise ValueError("The data and file are locked. Please copy them somewhere.")
    
        if self._image_type == "ROI":
            raise ValueError("Data is ROI and cannot be used for brain extraction")
    
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(dir="/tmp")
        
        # Copy input file to temporary directory
        temp_input_path = os.path.join(temp_dir, self._filename + self._endswith)
        shutil.copy2(self._path, temp_input_path)
        
        # Define output file paths in temporary directory
        csf_file = os.path.join(self._dir, self._filename + "_CSF_fast" + self._endswith)
        wm_file = os.path.join(self._dir, self._filename + "_WM_fast" + self._endswith)
        gm_file = os.path.join(self._dir, self._filename + "_GM_fast" + self._endswith)
        
        if (not os.path.isfile(csf_file) ) or ( not os.path.isfile(wm_file)) or  ( not os.path.isfile(gm_file)):
    
            try:
                # Perform FAST segmentation in the temporary directory
                subprocess.run(["fast", "-t", "2", temp_input_path], check=True, cwd=temp_dir)
            except subprocess.CalledProcessError as e:
                RuntimeError("Error during FAST segmentation:", e)
  
            
            # Define paths of segmented files in temporary directory
            csb_fluid_old_path            = [os.path.join(temp_dir,f) for f in os.listdir(temp_dir) if "pve_2" in f][0]
            white_matter_old_path         = [os.path.join(temp_dir,f) for f in os.listdir(temp_dir) if "pve_1" in f][0]
            grey_matter_old_path          = [os.path.join(temp_dir,f) for f in os.listdir(temp_dir) if "pve_0" in f][0]
            #mixeltype_old_path          = [os.path.join(temp_dir,f) for f in os.listdir(temp_dir) if "mixeltype" in f][0]
            #seg_old_path                = [os.path.join(temp_dir,f) for f in os.listdir(temp_dir) if "seg" in f][0]
        
            # Move segmented files back to the original directory with renaming
            shutil.move(csb_fluid_old_path, csf_file)
            shutil.move(white_matter_old_path, wm_file)
            shutil.move(grey_matter_old_path, gm_file)
            
            # Remove unnecessary files
            shutil.rmtree(temp_dir)
        else:
            print("fast allready run!")

        return (
            NiftiFileChecker(csf_file, locked=False),
            NiftiFileChecker(wm_file, locked=False),
            NiftiFileChecker(gm_file, locked=False),
        )

    
    def plot(self, slice_index=None, save=None,save_path=None , figsize=(15, 15),dpi = 50 ):
        """
        Plots the NIfTI file.
        
        Parameters:
        - slice_index (int or None): Index of the specific displayed slice to plot. If None, all displayed slices will be plotted as subplots. Default is None.
        - save (bool): Whether to save the plot as an image file. Default is None.
        - save_path (str): Path to save the plot image file. If None, the default path will be used. Default is None.
        - figsize (tuple): Figure size in inches. Default is (15, 15).
        - dpi (int): Dots per inch for the saved image file. Default is 50.
        
        Raises:
        - ValueError: If the image array is not available or if an invalid slice index is provided.
        
        Returns:
        - None
        
        """
        if self._image_array is None:
            raise ValueError("Image array is not available.")
            
        if not self._hasmask:

            if self.num_slices > self.max_displayed_slices:
                start_index = (self.num_slices - self.max_displayed_slices) // 2
                end_index = start_index + self.max_displayed_slices
                displayed_slices = range(start_index, end_index)
            else:
                displayed_slices = range(self.num_slices)
            
            if slice_index is None:
                # Plot displayed slices as subplots
                num_cols = int(np.ceil(np.sqrt(len(displayed_slices))))
                num_rows = int(np.ceil(len(displayed_slices) / num_cols))
                
                fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
                fig.suptitle(f"{self._filename}  {self._volume_ml} ml")
                
                plotted_slices = 0
                for i in displayed_slices:
                    row = plotted_slices // num_cols
                    col = plotted_slices % num_cols
                    
                    if np.any(self._image_array[..., i] != 0):
                        axs[row, col].imshow(self._image_array[..., i], cmap="gray")
                        axs[row, col].set_title(f"Slice {i+1}  {self._volume_ml_slices[i]} ml ")
                        axs[row, col].axis("off")
                        plotted_slices += 1
                
                plt.tight_layout()
                
                if self._locked:
                    print(" image locked cannot save the plot")
                
                if save and not self._locked:
                    if save_path:
                        save_dir, save_filename = os.path.split(save_path)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path_all = os.path.join(save_dir, save_filename + "_all.png")
                        plt.savefig(save_path_all)
                    else:
                        save_path_all = os.path.join(self._dir, self._filename + "_all.png")
                        plt.savefig(save_path_all)
                plt.show()
            else:
                # Plot a specific displayed slice
                if slice_index < 0 or slice_index >= len(displayed_slices):
                    logging.error(f"Invalid slice index. maximum slices {len(displayed_slices)}")
                    raise ValueError(f"Invalid slice index. maximum slices {len(displayed_slices)}")
                
                i = displayed_slices[slice_index]
                if np.any(self._image_array[..., i] != 0):
                    
                    plt.figure(figsize=figsize)
                    plt.imshow(self._image_array[..., i], cmap="gray")
                    plt.title(f"Slice {i+1}  of {self._filename}   {self._volume_ml_slices[i]} ml")
                    plt.axis("off")
                    
                    if self._locked:
                        print(" image locked cannot save the plot")
    
                    if save and not self._locked:
                        if save_path:
                            save_dir, save_filename = os.path.split(save_path)
                            os.makedirs(save_dir, exist_ok=True)
                            save_path_all = os.path.join(save_dir, save_filename + f"_slice_{i+1}.png")
                        else:
                            save_path_all = os.path.join(self._dir, self._filename + f"_slice_{i+1}.png")
                            
                        plt.savefig(save_path_all,dpi=dpi, bbox_inches='tight') 
                    plt.show()
                else:
                    print("Slice is empty and cannot be plotted.")
            
        else:
            print(f"{self._hasmask}")
            
            self._mask_array
        
            if self.num_slices > self.max_displayed_slices:
                start_index = (self.num_slices - self.max_displayed_slices) // 2
                end_index = start_index + self.max_displayed_slices
                displayed_slices = range(start_index, end_index)
            else:
                displayed_slices = range(self.num_slices)
            
            if slice_index is None:
                # Plot displayed slices as subplots
                num_cols = int(np.ceil(np.sqrt(len(displayed_slices))))
                num_rows = int(np.ceil(len(displayed_slices) / num_cols))
                
                fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
                fig.suptitle(f"{self._filename}  {self._volume_ml} ml")
                
                plotted_slices = 0
                for i in displayed_slices:
                    row = plotted_slices // num_cols
                    col = plotted_slices % num_cols
                    
                    if np.any(self._image_array[..., i] != 0):
                        axs[row, col].imshow(self._image_array[..., i], cmap="gray")
                        axs[row, col].imshow(self._mask_array[..., i], cmap="binary", alpha=0.5)
                        axs[row, col].set_title(f"Slice {i+1}  {self._volume_ml_slices[i]} ml ")
                        axs[row, col].axis("off")
                        plotted_slices += 1
                
                plt.tight_layout()
                
                if self._locked:
                    print(" image locked cannot save the plot")
                
                if save and not self._locked:
                    if save_path:
                        save_dir, save_filename = os.path.split(save_path)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path_all = os.path.join(save_dir, save_filename + "_all.png")
                        plt.savefig(save_path_all)
                    else:
                        save_path_all = os.path.join(self._dir, self._filename + "_all.png")
                        plt.savefig(save_path_all)
                plt.show()
            else:
                # Plot a specific displayed slice
                if slice_index < 0 or slice_index >= len(displayed_slices):
                    raise ValueError(f"Invalid slice index. maximum slices {len(displayed_slices)}")
                
                i = displayed_slices[slice_index]
                if np.any(self._image_array[..., i] != 0):
                    
                    plt.figure(figsize=figsize)
                    plt.imshow(self._image_array[..., i], cmap="gray")
                    plt.imshow(self._mask_array[..., i], cmap="binary" , alpha=0.5)
                    plt.title(f"Slice {i+1}  of {self._filename}   {self._volume_ml_slices[i]} ml")
                    plt.axis("off")
                    
                    if self._locked:
                        print(" image locked cannot save the plot")
    
                    if save and not self._locked:
                        if save_path:
                            save_dir, save_filename = os.path.split(save_path)
                            os.makedirs(save_dir, exist_ok=True)
                            save_path_all = os.path.join(save_dir, save_filename + f"_slice_{i+1}.png")
                        else:
                            save_path_all = os.path.join(self._dir, self._filename + f"_slice_{i+1}.png")
                            
                        plt.savefig(save_path_all,dpi=dpi, bbox_inches='tight') 
                    plt.show()
                else:
                    print("Slice is empty and cannot be plotted.")
            
            
    def info_all(self):
        for var_name, value in self.__dict__.items():
            print(f"{var_name}: {value}")       
            
    def print_header(self):
        print(self._nii.header)
            
        
    def info_methods(self):
        """
        Prints out the docstrings of all the methods in the class.
        """
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if not "_" in name:
                if method.__doc__:
                    print(f"Method: {name}")
                    print(f"  Docstring: {method.__doc__}")
                print()            
                
                
    def add_mask(self, mask_path):
        
        if self._locked:
            raise ValueError("The data and file are locked. Please copy them somewhere.")
            
        check_path_exists(mask_path)
        self._check_nifti_file_mask(mask_path)

        self._mask_path =  os.path.join(self._dir, self._mask_basename )
        nib.save(self._mask_nii, self._mask_path )
        
        self._hasmask = True
        
        print(f"Mask successfully added :  {self._mask_path} ")
        
        
    def nulling(self):
        
         if self._locked:
            raise ValueError("The data and file are locked. Please copy them somewhere.")
            
         null_image    =  (np.logical_not(self._mask_array[:,:,:])) * self._image_array[:,:,:]
         null_image_path = os.path.join(self._dir, self._filename +"_null_" +self._mask_filename + self._endswith)
         
         null_image_nib          =  nib.Nifti1Image(null_image, self.__affine)
         nib.save(null_image_nib, null_image_path )
         
         return  NiftiFileChecker(null_image_path,
                                  mni_mat=self._mni_mat,
                                  locked=False)
     
        
   
