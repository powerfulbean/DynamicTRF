import argparse

#python -u main.py -m ns -c CausalConv --nNonLinWin 2 --timeLags 0 700 --linStims onset env lex_sur --nonLinStims lex_sur  --batchSize 1 --randomSeed 42  --lr 0.001 0.01 --wd 0.001 --nBasis 21 --tarDir /scratch/jdou3/Mapping/Result/nnTRFPlus --epoch 200 --fTRFMode +-a,c --optimStr AdamW --lrScheduler cycle --studyName formal_test3

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folderName',
        type = str, 
        help = 'name of current study',
        default = 'dynamic_trf'
    )
    parser.add_argument('--tarDirRoot', type=str, default = './')
    parser.add_argument(
        '--mtrf_only',
        help = 'run the mtrf analysis only',
        action='store_true'
    )
    parser.add_argument(
        '--contextModel', 
        type = str, 
        help = "the name of the model to generate the parameter to transform the functional TRF",
        default = 'CausalConv'
    )
    parser.add_argument(
        '--nContextWin', 
        type = int, 
        help = "the window size of the model to generate the parameter to transform the functional TRF",
        default = 2
    )
    parser.add_argument(
        '--fTRFMode',
        type=str, 
        help = "mode of generating the parameter to transform the functional TRF, +- indicates 'a' can be both positive and negative",
        choices=['','a','b','a,b','a,b,c','a,c','+-a','b','+-a,b','+-a,b,c','+-a,c'],
        default = '+-a,b'
    )
    parser.add_argument(
        '--nBasis',
        type=int,
        help = "number of basis to represent the functional TRF",
        default=21,
    )
    parser.add_argument(
        '--timelag', 
        type=int,
        nargs=2,
        help = 'range of time lag (ms)',
        metavar=('tmin', 'tmax'),
        default=[0, 700] 
    )
    parser.add_argument(
        '--extraTimeLag', 
        type=int,
        help = 'the additional time lag length appended and preprended to the timelag (ms)',
        default=200 
    )
    parser.add_argument(
        '--nFolds',
        type=int,
        help = 'number of cross-validation folds',
        default=10
    )
    # parser.add_argument('--linStims', nargs='+', default = []) #the last of linStims will be used for template
    # parser.add_argument('--nonLinStims', nargs='+', default = [])
    parser.add_argument(
        '--epoch',
        help = "how many epochs of traning",
        type=int,
        default=100
    )
    # parser.add_argument(
    #     '--batchSize',
    #     type= int,
    #     help = 'how many runs of data being used in one round of backward propogation',
    #     default=1 
    # )
    parser.add_argument(
        '--wd',
        type=float,
        help = 'weight decay (regularization) parameter of dynamic TRF',
        default=0.01, 
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        nargs=2, 
        help = 'range of learning rate of dyamic TRF',
        metavar=('lr_min', 'lr_max'),
        default = [0.001,0.001]
    )
    # parser.add_argument('--foldList', nargs='+', type=int, default = [])
    parser.add_argument(
        '--optimizer', 
        type=str, 
        help = 'the optimizer to use',
        default = 'AdamW'
    )
    parser.add_argument(
        '--lrScheduler', 
        type = str, 
        help = 'the learning rate scheduler to use',
        default = 'cycle'
    )
    parser.add_argument('--device',default='cuda', type=str)
    parser.add_argument('--randomSeed',default=42, type=int)
    
    return parser.parse_args()