# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2017-10-23 13:13:42
# @Last Modified by:   richman
# @Last Modified time: 2017-10-23 13:15:53
import struct
import numpy as np
import sklearn.mixture.gaussian_mixture as gmm

def readhtk(fname):
    # Read header
    with open(fname, 'rb') as f:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(
            ">iihh", f.read(12))

        # Read data
        data = struct.unpack(
            ">%df" % (nSamples * sampSize // 4), f.read(nSamples * sampSize))
        return np.array(data).reshape(nSamples, sampSize // 4)


x = readhtk("./wav/chen_0004092_A.mfcc")
print(x.shape)

openfile = open('vad.gmm','r')
means = []
variances = []

for line in openfile:
    if '<MEAN>' in line:
        nextline = next(openfile)
        mean = np.array(nextline.split()).astype(float)
        means.append(mean)
    elif '<VARIANCE>' in line:
        nextline = next(openfile)
        variance = np.array(nextline.split()).astype(float)
        variances.append(variance)

means = np.array(means)
variances = np.array(variances)
print means.shape
print variances.shape

gmm
