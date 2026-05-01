from dynamic_trf.analysis.load_result import collect_results_across_folds, wilcoxon_fdr

path = r"/scratch/jdou3/box/dynamic_trf_2026_mod~lexsur_limTimeLag_smallregrange_mse"
collect_results_across_folds(path, 'testr')
