import glob
import sys 
import numpy as np
path_to_dir = '/project/RDS-FSC-QCL_KF-RW/kf/jobs_stats/'
search_phrases = ['Traceback (most recent call last):', 'Killed', 'Error:']
file_type = '*.o*'

#path_to_dir = sys.argv[1]
#search_phrse = sys.argv[2]
#file_type = sys.argv[3]


err_count=0
sch_files=0
err_filename_list = []
sch_filename_list=[]
for file_name in glob.glob(path_to_dir+file_type):
    with open(file_name) as f:
        contents = f.read()
    sch_files +=1
    sch_filename_list.append(file_name)

    for search_phrse in search_phrases:
        if search_phrse in contents:
            err_filename_list.append(file_name+'_with_'+search_phrse)
            err_count += 1

print("Total files in error: ", err_count )
print("Total files searched: ", sch_files)
np.savez(path_to_dir+'Search_Results', err_filename_list=err_filename_list, sch_filename_list=sch_filename_list)
print("Err file names saved" )
