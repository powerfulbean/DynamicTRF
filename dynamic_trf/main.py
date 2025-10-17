import numpy as np

from dynamic_trf.utils.io import load_dataset
from dynamic_trf.utils.args import get_arg_parser
from dynamic_trf.core import execute, NestedArrayList, NestedArrayDictList, Configuration

if __name__ == '__main__':


    ### prepare the dataset

    # load the paired stimuli and the responses

    # the stimuli contains continuous stimuli and discrete stimuli

    """
    the control_stims, target_stims, resps should be nested List of numpy array or StimDictArray (target_stims only)
        each item of the outer list corresponding to one subject, each item of the inner list corresponding to one trial
        the size of it is [# of subject * [# of trials * (n_samples, n_channels)]]
    """

    control_stims, target_stims, resps = load_dataset()
    control_stims: NestedArrayList
    target_stims:  NestedArrayDictList
    resps: NestedArrayList

    args = get_arg_parser()
    default_configs = vars(args).copy()
    user_configs = {
        'contextModel': 'CausalConv',
        'fTRFMode': '+-a,b' #real value amplitude scaling (a) amd time shifit (b)
    }
    
    configs = default_configs.copy()
    configs.update(user_configs)

    configs = Configuration(**configs)

    execute.run(control_stims, target_stims, resps, configs)


    
    otherParam = {}
    for k in args.__dict__:
        otherParam[k] = args.__dict__[k]
    ds = load_dataset(args.dataset, r'/scratch/jdou3/Mapping/dataset')
    stimuliDict = ds.stimuliDict
    # for k in stimuliDict:
    #     print(stimuliDict[k].keys())
    stimFeats = args.linStims.copy()
    ds.stimFilterKeys = stimFeats

    foldList = args.foldList
    test_mtrf = args.test_mtrf
    studyName = args.studyName
    randomSeed = args.randomSeed
    nFolds = args.nFolds

    testResults = []
    devResultsReduce = []
    testResultsReduce = []

    for i in execute.iterFold(nFolds) if len(foldList) == 0 else foldList:
        datasets = ds.nestedKFold(i,nFolds)
        if test_mtrf:
            (
                bestDevMetricsReduce, 
                testMetricsReduce,
                testMetrics, 
                oExpr
            ) = execute.test_mtrf(
                studyName, 
                datasets, 
                [i,10], 
                otherParam
            )
        else:
            (
                oTrainer,
                bestModel,
                modelMTRF,
                configs,
                bestDevMetricsReduce,
                oRun,
                oExpr
            ) = execute.train(
                studyName,
                {i:datasets[i] for i in ['train','dev']},
                randomSeed,[i,10],
                otherParam,
                args.epoch
            )
            (
                testMetricsReduce,
                testMetrics
            ) = execute.test(
                oTrainer,
                bestModel,
                modelMTRF,
                datasets['test'],
                oRun = oRun, 
                otherParam = otherParam
            )
        print(bestDevMetricsReduce)
        testResults.append(testMetrics)
        devResultsReduce.append(bestDevMetricsReduce)
        testResultsReduce.append(testMetricsReduce)