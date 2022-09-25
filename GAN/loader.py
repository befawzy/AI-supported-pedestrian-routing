from torch.utils.data import DataLoader, dataloader

from trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    '''
        Function to generate mini-batches for the training loop. The returned object can run multiple workers in parallel.

        Parameters
        ----------
        path : str
            path to raw data file(s) 

        Returns
        -------

        dset: class object
            object containing trajectories of raw data. Built by the defined class TrajectoryDataset

        loader : class object
            provides iterable mini-batches for the training (and plotting) loops 
        '''
    dset = TrajectoryDataset(
        path,
        obs_len=6,
        pred_len=6,
        skip=1)

    loader = DataLoader(
        dset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate)
    return dset, loader

