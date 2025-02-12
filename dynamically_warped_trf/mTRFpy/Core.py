# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:50:28 2020

@author: Jin Dou
"""

import numpy as np
from . import Protocols as prtcls
# from memory_profiler import profile
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.csc import csc_matrix
from scipy.sparse import hstack

DEBUG = False

TypeEnum = tuple(['multi','single'])
ErrorEnum = tuple(['mse','mae'])
oPrtclsData = prtcls.CProtocolData()
oCuda = None
sparseFlag = False

def matMul(x,y):
    '''
    calculat the matrix product of two arrays
    x: left matrix
    y: right matrix
    
    '''
    if sparseFlag:
        return x @ y
    else:
        return np.matmul(x,y)

def calCovariance(x,y):
    '''
    calculat the covariance of two matrices
    x: left matrix
    y: right matrix
    
    #if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
    '''
    # oPrtclsData(x,y)
    # print(x.shape,y.shape)
    if sparseFlag:
        return x.T @ y
    else:
        return np.matmul(x.T,y)
    
def calSelfCovariance(x):
    '''
    calculat the covariance of matrix itself
    x: input matrix
    
    #if the input for x and y are both 1-D vectors, they will be reshaped to (len(vector),1)
    '''
    # oPrtclsData(x,y)
    # print(x.shape,y.shape)
    if sparseFlag:
        return x.T @ x
    else:
        return np.matmul(x.T,x)

def genLagMat(x,lags,Zeropad:bool = True,bias =True): #
    '''
    build the lag matrix based on input.
    x: input matrix
    lags: a list (or list like supporting len() method) of integers, 
         each of them should indicate the time lag in samples.
    
    see also 'lagGen' in mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox
    
    
    #To Do:
       make warning when absolute lag value is bigger than the number of samples
       implement the zeropad part
    '''
    # oPrtclsData(x)
    nLags = len(lags)
    nSamples = x.shape[0]
    nVar = x.shape[1]
    # lagMatrix = np.zeros((nSamples,nVar*nLags))
    # print(type(x),isinstance(x, csr_matrix))
    if isinstance(x, csc_matrix):
        lagMatrix = lil_matrix((nSamples,nVar*nLags))
        # x = x.toarray()
    else:
        lagMatrix = np.zeros((nSamples,nVar*nLags))
    # print(type(lagMatrix),lagMatrix)
    for idx,lag in enumerate(lags):
        colSlice = slice(idx * nVar,(idx + 1) * nVar)
        if lag < 0:
            lagMatrix[0:nSamples + lag,colSlice] = x[-lag:,:]
        elif lag > 0:
            lagMatrix[lag:nSamples,colSlice] = x[0:nSamples-lag,:]
        else:
            lagMatrix[:,colSlice] = x
    
    if not Zeropad:
        lagMatrix = truncate(lagMatrix,lags[0],lags[-1])
        
    if bias:
        if isinstance(x, csc_matrix):
            ones = lil_matrix((lagMatrix.shape[0],1))
            ones[:] = 1
            lagMatrix = hstack([ones,lagMatrix])
        else:
            lagMatrix = np.concatenate([np.ones((lagMatrix.shape[0],1)),lagMatrix],1);

#    print(lagMatrix.shape)    
    if sparseFlag:
        lagMatrix = csr_matrix(lagMatrix)
    
    return lagMatrix

def genRegMat(n:int, method = 'ridge'):
    '''
    generates a sparse regularization matrix of size (n,n) for the specified method.
    see also regmat.m in mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox
    '''
    regMatrix = None
    if method == 'ridge':
        regMatrix = np.identity(n)
        regMatrix[0,0] = 0
    elif method == 'Tikhonov':
        regMatrix = np.identity(n)
        regMatrix -= 0.5 * (np.diag(np.ones(n-1),1) + np.diag(np.ones(n-1),-1))
        regMatrix[1,1] = 0.5
        regMatrix[n-1,n-1] = 0.5
        regMatrix[0,0] = 0
        regMatrix[0,1] = 0
        regMatrix[1,0] = 0
    else:
        regMatrix = np.zeros((n,n))
    return regMatrix

# @profile
def calOlsCovMat(x,y,lags,Type = 'multi',Zeropad = True):
    assert Type in TypeEnum
    
    if not Zeropad:
        y = truncate(y,lags[0],lags[-1])
    
    if Type == 'multi':
        xLag = genLagMat(x,lags,Zeropad)
        if oCuda is None:
            Cxx = calSelfCovariance(xLag)
            Cxy = calCovariance(xLag,y)
        else:
            Cxx = oCuda.calSelfCovariance(xLag)
            Cxy = oCuda.calCovariance(xLag,y)
    return Cxx, Cxy

def calOlsCovMat_PartFeatsLag(x,y,lags,xIdxNoLag,Type = 'multi',Zeropad = True):
    assert Type in TypeEnum
    
    if not Zeropad:
        y = truncate(y,lags[0],lags[-1])
    
    nFeat = x.shape[1]
    fullList = [i for i in range(nFeat)]
    xIdxToLag = np.array([i for i in fullList if i not in xIdxNoLag])
    xIdxNoLag = np.array(xIdxNoLag)
    
    if Type == 'multi':
        xLag = genLagMat(x[:,xIdxToLag],lags,Zeropad)
        xNoLag = x[:,xIdxNoLag]
        xLag = np.concatenate([xNoLag,xLag],axis = 1)
        if oCuda is None:
            Cxx = calSelfCovariance(xLag)
            Cxy = calCovariance(xLag,y)
        else:
            Cxx = oCuda.calSelfCovariance(xLag)
            Cxy = oCuda.calCovariance(xLag,y)
    return Cxx, Cxy

def msec2Idxs(msecRange,fs):
    '''
    convert a millisecond range to a list of sample indexes
    
    the left and right ranges will both be included
    '''
    assert len(msecRange) == 2
    
    tmin = msecRange[0]/1e3
    tmax = msecRange[1]/1e3
    return list(range(int(np.floor(tmin*fs)),int(np.ceil(tmax*fs)) + 1))

def Idxs2msec(lags,fs):
    '''
    convert a list of sample indexes to a millisecond range
    
    the left and right ranges will both be included
    '''
    temp = np.array(lags)
    return list(temp/fs * 1e3)

def truncate(x,tminIdx,tmaxIdx):
    '''
    the left and right ranges will both be included
    '''
    rowSlice = slice(max(0,tmaxIdx),min(0,tminIdx) + len(x))# !!!!
    output = x[rowSlice]
    return output

def pearsonr(x,y):
    x,y = oPrtclsData(x,y)
    nObs = len(x)
    sumX = np.sum(x,0)
    sumY = np.sum(y,0)
    sdXY = np.sqrt((np.sum(x**2,0) - (sumX**2/nObs)) * (np.sum(y ** 2, 0) - (sumY ** 2)/nObs))
    
    r = (np.sum(x*y,0) - (sumX * sumY)/nObs) / sdXY
    return r
    
def error(x,y,error = 'mse'):
    assert error in ErrorEnum
    x,y = oPrtclsData(x,y)
    ans = None
    if error == 'mse':
        ans = np.sum(np.abs(x - y)**2, 0)/len(x)
    elif error == 'mae':
        ans = np.sum(np.abs(x - y),0)/len(x)
    return ans