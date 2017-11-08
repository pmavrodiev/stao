from multiprocessing import Pool
import pandas as pd


def apply_parallel(dfGrouped, func, ncpus, chunk_size):
    with Pool(ncpus) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped], chunksize=chunk_size)
    concatenated_pd = pd.concat(ret_list)
    return concatenated_pd


def filter(n_nearest, d):
    # if n_nearest is None:
    #    return d
    # dd = d.sort_values(by=['FZ'], ascending=True)
    # dd["neighbour_index"] = range(1, len(dd) + 1)
    # return dd
    return d.sort_values(by=['FZ'], ascending=True).head(n_nearest)
