import glob
import sys 
import numpy as np

path_to_dir = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/*/' # sys.argv[1]
file_type = '*.npz'

err_count=0
sch_files=0
err_list = []
sch_filename_list=[]
exp_list = []
unloaded_files = []
unload_count = 0

for filename in glob.glob(path_to_dir+file_type):

    try:
        obj_ = np.load(filename)
    
    except:
        print('%s could not be loaded' %(filename))
        unloaded_files.append(filename)
        unload_count +=1
        continue
    
    sch_filename_list.append(filename)
    sch_files +=1

    for data_key in obj_.files:

        try:
            data_array = obj_['%s'%(data_key)]
        except:
            print('%s in %s could not be loaded' %(data_key, filename))
            continue 
        
        try:
            if isinstance(data_array, basestring):
                continue
            if np.any(np.isnan(data_array)) == True or np.any(np.isinf(data_array)) == True:
                err_list.append('Inf-Nan == True in %s in %s' %(data_key, filename))
                err_count +=1
            elif np.any(np.isfinite(data_array)) != True:
                err_list.append('isFinite == False in %s in %s'%(data_key, filename))
                err_count +=1
        
        except Exception as inst:
            exp_list.append("Exception Raised %s in %s in %s"%(type(inst), data_key, filename))

        continue

print('Total NPZ files: ', sch_files)
print('Total Data Inf, Nan, not Finite Instances: ', len(err_list))
print('Total Exception Instances: ', len(exp_list))
print('Total Unloaded Files: ', unload_count)
print('...')
print('...')
print('The following files could not be loaded')
print(unloaded_files)
print('...')
print('...')

np.savez('NPZ_Results', exp_list=exp_list,unloaded_files=unloaded_files, err_list=err_list, sch_filename_list=sch_filename_list,sch_files=sch_files, err_count=err_count)

print('COMPLETE')
