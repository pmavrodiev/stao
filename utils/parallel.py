from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

"""
    Hektars are distributed across threads (apply_parallel). Each thread then does locally:
        1. Grouping by store type (group_by_store_type)
        2. Prune each resulting group of store types (prune)

    This parallelization approach proved to be fastest, especially compared to finer grained distribution,
    such as distributing also the pruning per store type.
"""


def apply_parallel(dfGrouped, func, ncpus, chunk_size):
    with Pool(ncpus) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped], chunksize=chunk_size)
    concatenated_pd = pd.concat(ret_list)
    return concatenated_pd


def prune(d):
    def prune_fz(dd):
        if len(dd) == 1:
            return dd
        else:
            idx_to_return = dd['RLAT'] == np.max(dd['RLAT'])
            return dd.loc[idx_to_return]

    d_sorted = d.sort_values(by='fahrzeit', ascending=True)
    d_pruned = d_sorted.groupby(['fahrzeit']).apply(prune_fz)

    tmp = [x for x in d_pruned.index.names]
    if (tmp[0] == 'fahrzeit'):
        tmp[0] = 'a'
    d_pruned.index.names = tmp
    d_pruned = d_pruned.reset_index().set_index(keys=['hektar_id', 'type'])

    for column_to_delete in ['a', 'level_1', 'index']:
        if column_to_delete in d_pruned:
            del d_pruned[column_to_delete]

    # with open(config["output"]["pruned_filiale"], 'a') as f:
    #     f.write('BEFORE PRUNING\n')
    #     d_pruned.to_csv(f, header=True)

    while True:
        diffs = np.diff(d_pruned['RLAT'])
        return_idx = (diffs > 0)
        return_idx = np.insert(return_idx, 0, True)  # always take the first entry
        d_pruned = d_pruned.loc[return_idx]
        if len(d_pruned) == 1 or all(return_idx):
            break

    # with open(config["output"]["pruned_filiale"], 'a') as f:
    #    f.write('AFTER PRUNING\n')
    #    d_pruned.to_csv(f, header=True)

    return d_pruned


def group_by_store_type(d):
    ret = d.reset_index().groupby('type').apply(prune)
    ret.index.names = ['a', 'hektar_id', 'type']
    ret = ret.reset_index()
    del ret['a']
    return ret
