#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:32:31 2023

@author: temuuleu
"""

import os
from dotenv import load_dotenv
from colorama import Fore
from config.singleton import Singleton

load_dotenv(verbose=True)


class Config(metaclass=Singleton):
    
    def __init__(self) -> None:
        self.masterfile_dir_path       =  os.getenv("masterfile_dir_path")
        self.masterfile_file_path      =  os.getenv("masterfile_file_path")
        self.standard_space_flair      =  os.getenv("standard_space_flair")
        self.standard_space_ventrikel  =  os.getenv("standard_space_ventrikel")
        self.standard_mask             =  os.getenv("standard_mask")
        self.data_test_dir_path        =  os.getenv("data_test_dir_path")
        self.data_out_test_dir_path    =  os.getenv("data_out_test_dir_path")
        self.train_bids_dir            =  os.getenv("train_bids_dir")
        
        self.train_bids_dir_master_1   =  os.getenv("train_bids_dir_master_1")
        
        self.TEST_DATA_DIR             =  os.getenv("TEST_DATA_DIR")

        self.trainingpts               =  os.getenv("trainingpts")
        self.selectpts                 =  os.getenv("selectpts")
        self.nonlespts                 =  os.getenv("nonlespts")
        self.outputdir                 =  os.getenv("outputdir")

               
def check_master_file_list_path() -> None:
    """Check if the MASTERFILE_PATH is set in config.py or as an environment variable."""
    cfg = Config()
    if not cfg.master_file_path:
        print(
            Fore.RED
            + "Please set your MASTERFILE_PATH in .env or as an environment variable."
        )
        exit(1)
