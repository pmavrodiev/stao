| /userdata/pmavrodi/Projekte/Stao/output
|
|
|--- *enriched_pruned_pd.pkl* : Pickled pandas data frame containing the data for all stores
|       and all hektars, including the LATs and RLATs, AFTER the pruning step was executed.
|       The RLATs were computed with the initial value of the parameters used by MP: a=0.1 and b=0.001
|       The pickled object can be used to try out different parameter values, without having to prune
|       after each combination - this saves a lot of time. A caveat here is that the results will slightly
|       deviate from the ideal case (i.e. pruning after each param. combination), because depending on how much
|       the parameters were changed, RLATs may change in a way that violates the pruning criteria, hence pruning is necessary.
|       These deviation turn out to be small and hence inconsequential in practice. Therefore an efficient approach
|       to tune parameters is to find an optimum using this cache and then re-run the analysis with
|       the tuned parameters and pruning.
|
|
|--- *enriched_PRUNED_original_pd.pkl* : Should be the same as *enriched_PRUNED_original_pd.pkl*
|
|
|--- *parameter_sweep_mit_pruning.log* : Log of the parameter sweep over a and b.
|     The pruning step was executed after each parameter combination. The most important
|     information in the log is the value of the cost function (TOTAL LINEAR SQUARE ERROR)
|     for each parameter combination
|
|--- *parameter_sweep_mit_pruning_VFL_FOOD_FRISCH.log* : Same as *parameter_sweep_mit_pruning.log*
|     but instead of the total Verkaufsflaeche, the Food and Frisch Verkaufslaeche was used to calculate the LATs.
|      Note that if not explicitly specified, the total Verkaufsflaeche was used.
|
|--- *parameter_sweep_ohne_pruning.log* - Again a parameter sweep, but without pruning.
|
|
|--- *umsatz_potential_param_sweep_*.tgz*
|     An archive containing the Umsatz predictions for each parameter combination. The filename indicates whether
|     pruning took place and which Verkaufsflaeche was used to compute the LATs.
|     It also points to the corresponding log file
|
|--- *umsatz_potential_pd.txt* - The Umsatz predictions for the original set-up as used by MP. That is a=0.1 and b=0.001


