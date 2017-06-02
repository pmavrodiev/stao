
### Input Data


### settings.cfg

#### [global]

 - **cache_enabled**

      If True the input data will be read from the cached pickled
      objects specified in the [cache_config] section.

      If False the input data will be read from the files specified in
      the [inputdata] section. Once read, the data will be cached into
      the pickled objects specified in [cache_config]

 - **single_store**

      If the analysis should be run for a single store only (e.g.
      for testing) then specify the store's name, e.g. SM_MIG_49972_11914

      If empty, the analysis is run over all stores.

 - **prune**

      If True execute the pruning step of v1.1. Otherwise not

#### [parallel]

 This section defines the parallelization settings for the pruning step.

 - **cpu_count**

    Number of processes to be used.
    This is the parameter for the *multiprocessing.pool.Pool* constructor
    from the Python multiprocessing module.

 - **chunk_size**

    The number of chunks into which the data should be split.
    This is the *chunk_size* parameter for the *multiprocessing.pool.Pool.map*
    function in the Python multiprocessing module.

 The Pruning is executed as follows.

 The first step is the distribution phase. The data are grouped according to *hektar*. Each hektar group
 is then split into *chunk_size* chunks, which are then submitted to the
 pool of *cpu_count* processes. Hence each process receives at a time a subset
 of the data from a given hektar group.

 The second step is the actual pruning logic and is executed locally.
 Each process filters out the stores that do not satisfy the pruning
 criterion (read model documentaton for the criterion). Since the data
 that the process has is for the same hektar, the local pruning is correct.

#### [calibration]

 This section defines settings to configure the automated parameter
 calibration. So far only the RLAT function has been parametrized.

 - **a_start** and **b_start**

    The 2 free parameters in the RLAT function.

 - **T**

    Maximum number of time steps for the gradient descent algorithm.

 - **delta_convergence**

    The convergence criterion for the gradient descent algorithm.
    The algorithm ends when the error function becomes smaller than this value.

 - **use_pruned_cache**

    If True, read a snapshot of the analyzed data from the cache specified
    in [output][output_pickle] and proceed with calculating the Umsatz prediction.

    If False, proceed with calculating the RLATs using *a_start* and *b_start*.
    Afterwards, if [global][prune] is True execute the pruning step and cache
    the pruned data in [output][output_pickle]. If [global][prune] is False simply
    cache the data in [output][output_pickle].

    The most important information that the snapshot contains is the calculated RLATs
    and the post-pruned store distribution. Therefore it makes sense to create such
    a cache with pruning enabled, so that subsequent repeated analysis can quickly
    load proceed with parameter calibration without doing the relatively time-consuming
    pruning step.

 - **direct_output**

    Skip over the parameter calibration and compute directly the Umsatz prediction.
    The store information used is either from the pruned cache (controlled by *use_pruned_cache*)
    or is calculated anew using *a_start* and *b_start* and [global][prune]

    The Umsatz predictions are written into [output][output_csv]

#### [parameter_sweep]

