from datetime import datetime, timedelta
import pathlib
from time import time

import numpy as np

from fv3dataset import FV3Dataset

if __name__ == '__main__':

    local_zarr_path = '/lustre/Linlin.Cui/mlsfs/data/UFS'
    data_dict = {
        "train": range(1996, 2017),
        "test": range(2017, 2019),
        "out_of_sample": range(2019, 2023),
    }
    
    for k, v in data_dict.items():
        outdir = f'{local_zarr_path}/{k}'
        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)


        t0 = time()
        for year in v:
            print(year)
            fv3 = FV3Dataset(outdir, year, initilize=True)
            fv3.process_data_with_wgrib(datetime(year, 1, 1, 6))

        print(f'Time: {time()-t0} sec')
