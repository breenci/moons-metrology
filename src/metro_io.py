"""
This module contains functions for reading and writing data from metrology and 
FPU output files.    
"""
import pandas as pd

def read_metro_out(fn):
    """Read the metrology output"""
    colnames = ['ID', 'R', 'Theta', 'Parity']
    metro_df = pd.read_csv(fn, sep=' ', header=None, names=colnames)
    return metro_df


def read_FPU_out(fn):
    """Read the FPU output"""
    colnames = ['ID', 'R', 'Theta', 'Parity']
    fpu_df = pd.read_csv(fn, sep='\t', header=None, names=colnames, skiprows=1,
                         index_col=False, usecols=[0, 1, 2, 3])
    return fpu_df


def read_fibmap(fn):
    """Read the fibre map"""
    colnames = ['ID', 'X', 'Y', 'Z']
    fibmap_df = pd.read_csv(fn, sep=' ', header=None, names=colnames, skiprows=18,
                            usecols=[0, 13, 14, 15])
    
    return fibmap_df


def read_metro_raw(fn):
    """Read the raw metrology output"""
    colnames = ['ID', 'X', 'Y', 'Z', 'errX', 'errY', 'errZ', 'S']
    raw_df = pd.read_csv(fn, sep=' ', header=None, names=colnames)
    return raw_df


def read_obc(fn):
    """Read the OBC file"""
    colnames = ["ID", "X", "Y", "Z", "X_err", "Y_err", "Z_err", "Rays", 
                "check1", "check2", "check3"]
    obc_df = pd.read_fwf(fn, header=None, names=colnames)
    return obc_df