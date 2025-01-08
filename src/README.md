Network design algorithm based on Iterative Reweighted L1 norm minimization
Dataset scripts:    
    - load_dataset_catalonia.py Merges multiple air pollution datasets into a single file
    - dataset_preprocessing.py cleans air pollution dataset from missing and other techniques
    - NOAA_dataset.py loads NOAA SST files, perform image preprocessing technique and create snapshots matrix for the method

Methods scripts:
    - sensor_placement.py performs typical sensor placement algorithms (D-optimal, rankMax, etc)

Main scripts:
    - IRNet_AirPollution.py/network_planning.py performs main algorithm on Air pollution dataset
    - Dopt_placement.py performs sensor selection on Air pollution dataset using Joshi-Boyd sensor selection algorithm.
	Used to compare with results of network_planning algorithm
    - network_design_SST.py performs main algorithm on SST dataset and applies large-scale deployment method
    - IRNET_designNetwork.sh executes network_design_SST.py

IRNET_designNetwork.sh usage:
    - epsilon and num_it are IRNet parameters for convergence
    - batch_rows and batch_cols are minibatch size of image
        batch_rows*batch_cols determine the batch_size
        The number of batch_rows and batch_cols should be multiple of the SST image (original, windowed or downsampled,etc)
        Values used:
            - batch_size = 100 / batch_rows = 10, batch_cols = 10
            - batch_size = 80  / batch_rows = 8,  batch_cols = 10
            - batch_size = 50  / batch_rows = 10, batch_cols = 5
            - batch_size = 25  / batch_rows = 5,  batch_cols = 5
            - batch_size = 20  / batch_rows = 5,  batch_cols = 4
            - batch_size = 10  / batch_rows = 5,  batch_cols = 2
            - batch_size = 4   / batch_rows = 2,  batch_cols = 2  
    - b_init in case of not starting from inital batch

