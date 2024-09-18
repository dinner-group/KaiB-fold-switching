import os 
import subprocess as sp

temps = [2.5 + 0.5*(i) for i in range(16)]

original_dfiles = ['full_tuned_cis.dat','full_tuned_trans.dat']


for i in temps:

    print('Making barrier =' + str(i))
    
    fileprefix = 'barrier_'+str(i)

    for iso in original_dfiles:

        filename = fileprefix + iso[iso.rfind('_'):]

        command = 'sed s/\' 3 \'/\' ' + str(i) +  ' \'/g ' + iso 
        command = command + ' > ' + filename
        print(command)
        sp.check_call(command,shell = True)


