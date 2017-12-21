# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2017-10-23 13:13:42
# @Last Modified by:   richman
# @Last Modified time: 2017-10-23 13:15:53
import struct
import numpy as np
import sys
from sklearn import mixture

def readhtk(fname):
    # Read header
    with open(fname, 'rb') as f:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(
            ">iihh", f.read(12))

        # Read data
        data = struct.unpack(
            ">%df" % (nSamples * sampSize // 4), f.read(nSamples * sampSize))
        return np.array(data).reshape(nSamples, sampSize // 4)


x = readhtk("./wav/chen_0004092_B.mfcc")

openfile = open('vad.gmm','r')
means = []
variances = []
weights = []

for line in openfile:
    if '<MEAN>' in line:
        nextline = next(openfile)
        mean = np.array(nextline.split()).astype(float)
        means.append(mean)
    elif '<VARIANCE>' in line:
        nextline = next(openfile)
        variance = np.array(nextline.split()).astype(float)
        variances.append(variance)
    elif '<MIXTURE>' in line:
        [str, num, weight] = line.split()
        weights.append(float(weight))


variances.remove(variances[0])
means = np.array(means)
means_speech = means[0:128, :]
means_sil = means[128:256,:]
means_noise = means[256:384,:]

variances = np.array(variances)
variances_speech = variances[0:128, :]
variances_sil = variances[128:256,:]
variances_noise = variances[256:384,:]

weights = np.array(weights)
weights_speech = weights[0:128]
weights_sil = weights[128:256]
weights_noise = weights[256:384]
#print sum(weights_speech)   # 0.999961334
#print sum(weights_sil)      # 0.999860007
#print sum(weights_noise)    # 0.999980227

def gaussian(x, mu, sigma):
    """
    multi-dimentional guassian distribution
    """
    ans = 0.0
    for i in range(39):
        ans = ans + np.exp(-0.5*(x[i]-mu[i])*(x[i]-mu[i])/sigma[i])
    return ans

class GaussianMixture():
    def __init__(self, __means, __variances, __weights):
        self.__means = np.array(__means)
        self.__variances = np.array(__variances)
        self.__weights = np.array(__weights)
    def __str__(self):
        return "means = %s\nvariances = %s\nweights = %s" % (self.__means,self.__variances, self.__weights)
    def predict(self, x):
        ans = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            for i in range(128):
                ans[j] = ans[j] + self.__weights[i] * gaussian(x[j], self.__means[i], self.__variances[i])
            if j % (x.shape[0] / 10) == 0:
                print >>sys.stderr, "%s %%" % (j / (x.shape[0] / 10)*10)

        return ans


gmm_speech = GaussianMixture(means_speech, variances_speech, weights_speech)
gmm_sil = GaussianMixture(means_sil, variances_sil, weights_sil)
gmm_noise = GaussianMixture(means_noise, variances_noise, weights_noise)

#x = x[0:1000,:]
prob_speech = gmm_speech.predict(x)
prob_sil = gmm_sil.predict(x)
prob_noise = gmm_noise.predict(x)

#print prob_speech
#print prob_sil
#print prob_noise

n = x.shape[0]
last = 0
status = 0
for i in range(n):
    if i*10 < last:
        continue
    if status == 0:
        if prob_speech[i] > prob_sil[i] and prob_speech[i] > prob_noise[i]:
            print last, i*10 + 25, "sil"
            last = i*10 + 25
            status = 1
    elif status == 1:
        if not(prob_speech[i] > prob_sil[i] and prob_speech[i] > prob_noise[i]):
            print last, i*10 + 25, "speech"
            last = i*10 + 25
            status = 0
    pass
