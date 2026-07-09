from dynamic_trf.analysis.load_result import collect_results_across_folds, wilcoxon_fdr

path = r"/scratch/jdou3/box/dynamic_trf_2026_mod~lexsur_05012026_ctx1"
collect_results_across_folds(path, 'testr')

[ 9, 13, 14, 15, 16, 20, 21, 27, 29]