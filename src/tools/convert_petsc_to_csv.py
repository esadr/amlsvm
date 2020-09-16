import os 
import sys
import timeit
import pdb
import math
import numpy as np
import pickle
import pandas as pd 
import scipy.sparse as ss
import argparse


dirs = os.environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
try:
    import PetscBinaryIO
except: 
    print('Failed to import Petsc, check PETSC_DIR environment!')
    
io = PetscBinaryIO.PetscBinaryIO()


#print('try to set the temp directory on Palmetto ...')
#dir_tmp = os.listdir('/local_scratch')
#for item in dir_tmp:
    #if 'pbs' in item:
        #tmpdir= f'/local_scratch/{item}/' 
        #break
#else:
    #raise Exception('This is not the Palmetto cluster!')
    ## I guess for other machines, I can use /tmp 
#print(f'temp directory is set to {tmpdir}')

def main(args):
    # # Read the PETSc matrix
    #path = '/zfs/safrolab/users/esadrfa/datasets/medium_size/'
    path = args.path
    #ds_name = 'buzz' 
    ds_name = args.ds_name 
    dt_fname = f'{ds_name}_zsc_data.dat'
    lbl_fname = f'{ds_name}_label.dat'

    dt_mat = io.readBinaryFile(os.path.join(path, dt_fname), 
                            mattype='scipy.sparse')[0]
    lbl_vec = io.readBinaryFile(os.path.join(path, lbl_fname), 
                            mattype='scipy.sparse')[0]

    print('loaded files shapes:', dt_mat.shape, lbl_vec.shape)

    dt_arr = dt_mat.toarray()
    lbl_arr = np.array(lbl_vec.tolist())
    lbl_arr = lbl_arr.reshape((-1,1))
    data_lbl = np.hstack([lbl_arr, dt_arr])
    print(f'final data shape:{data_lbl.shape}')

    csv_format = ['%.18e' for i in range(data_lbl.shape[1])]
    csv_format[0] = '%d'
    out_file = os.path.join(path, f'{ds_name}.csv')
    
    np.savetxt(out_file,
          data_lbl, fmt=csv_format, delimiter=',',
          header='')


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='path', 
                        required=True)
    parser.add_argument('-f', action='store', dest='ds_name', 
                        required=True)
    
    args = parser.parse_args()    
    print(args)
    main(args)
    
