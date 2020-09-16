







from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_svmlight_file


data_path = "/local_scratch/pbs.7449306.pbs02/"
final_path = '.'
mem = Memory(data_path)

@mem.cache
def get_data():
    data = load_svmlight_file("mnist8m.scale")
    return data[0], data[1]

X, y = get_data()


# export the data using pickle
with open(f'{final_path}/mnist_scale_data.p', 'wb') as fh:         
pickle.dump(X, fh, pickle.HIGHEST_PROTOCOL)

with open(f'{final_path}/mnist_multiclass_lable.p', 'wb') as fh:
pickle.dump(y, fh, pickle.HIGHEST_PROTOCOL)



mat = X
mat = check_array(mat, accept_sparse='csr')
print(f'type(mat):{type(mat)} after check_array')
print(f'mat.shape:{mat.shape}')
print(f'mat.nnz:{mat.nnz}')

# write to PETScBinary format
out_filename = f"{final_path}/mnist_zsc_data.dat"

print('writeCsrMatrix2PETSc into: %s'\
        % out_filename , flush=True)
with open(out_filename + '.dat', 'w') as result_file:
    PetscBinaryIO.PetscBinaryIO().writeMatSciPy(result_file, mat)


# build the labels for all classes    
lbl = dict()
for i in range(10):
    bc = copy.deepcopy(y)
    #set the current class to -2
    
    bc[bc != i] = -1
    bc[bc == i] = 1
    lbl[i] = bc
    
# check the labels 
for i in range(10):
    print(np.unique(lbl[i]), np.sum(lbl[i]))

#export the labels to PETScBinary format
for i in range(10):
    vec = lbl[i].view(PetscBinaryIO.Vec)
    PetscBinaryIO.PetscBinaryIO().writeBinaryFile(f'mnist_c{i}_label.dat', [vec,])
    

print('everything is finished successfully!')
        
        
    
