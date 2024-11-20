from datetime import datetime, timedelta
import argparse
import pathlib
from time import time

import numpy as np

from fv3dataset import FV3Dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate UFS-replay zarr dataset")
    parser.add_argument("year", help="Which year to get the data")
    args = parser.parse_args()
    year = int(args.year)
    print(year)

    outdir = '/lustre/Linlin.Cui/mlsfs/data/UFS/train'

    fv3 = FV3Dataset(outdir, year)

    t0 = time()

    datevectors = np.arange(
        datetime(year, 1, 1, 12),
        datetime(year+1, 1, 1, 6),
        timedelta(hours=6),
    ).astype(datetime)
    for date in datevectors:
        print(date)
        fv3.process_data_with_wgrib(date)

    print(f'Time: {time()-t0} sec')
