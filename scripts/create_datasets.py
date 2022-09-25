from numpy.lib.npyio import genfromtxt
import math
import numpy as np

def create_dataset(path, xmin, xmax, ymin, ymax, output_name):
    '''
        Function to create subsets of data from an entire dataset (of momenTUM), that 
        represent chosen types of behavior. The function also split the subsets into training and validation 
        sets, and saves both as text files, to be ready to be fed to the neural network for training.


        Parameters
        ----------
        path : str
            path to raw data file(s)

        xmin : float 
            min x coordinate

        xmax : float
            max x coordinate

        ymin : float 
            min y coordinate

        ymax : float
            max y coordinate

        output_name: str
            desired output file name  

    '''
    data = np.genfromtxt(path)
    # first select only the data in the specified area with the given coordinates
    new_data=[]
    for i in range( len(data[:,2])):
        new_data.append((data[i,2] >= xmin and data[i,2] <= xmax and data[i,3] >= ymin and data[i,3] <= ymax))

    data =data[new_data]
    # sort the selected data by their id number , i.e. all time steps for id number 1 first then id no. 2 and so on.
    # This is done so that we have only sequential data for each id sorted out.
      
    Pedsnum, count = np.unique(data[:,1],return_counts=True)
    # create a new variable to store the sorted data
    result= None
    for i in Pedsnum:
        ped_pos= data[data[:,1]== i]
        for j in range(1,len(ped_pos),1):
            if ped_pos[j,0] - ped_pos[j-1,0] != 10:
                break
        ped_pos = ped_pos[:j, :]
        result = np.append(result, ped_pos)
    result = np.delete(result, 0)
    result = result.reshape((int((result.shape[0])/4),4))

    result = result[result[:,0].argsort()]
    
    train = result[:math.floor(0.75*len(result)) ]
    val = result[math.floor(0.75*len(result)) :]
    #save the 
    np.savetxt("train_" + output_name, train, delimiter='   ')
    np.savetxt("val_" + output_name, val, delimiter='   ')
