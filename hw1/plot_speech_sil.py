import librosa
import matplotlib.pyplot as plt
import os
import numpy as np

wavstart = 0
wavLen = 1000000

def plot_wav(wavefile):
#	wavLen = wavefile[0].shape[0]
	plt.plot(wavefile[0][wavstart:wavstart+wavLen], c = 'r')

plt.subplot(3,1,1)
plot_wav(librosa.load('en_4092_a.wav', sr = None))

os.system('python cutwav.py en_4092_b.wav en_4092_b.trns b_wav.wav')
os.system('python cutsil.py en_4092_b.wav en_4092_b.trns b_sil.wav')

wavefilea = librosa.load('b_wav.wav', sr = None)
wavefileb = librosa.load('b_sil.wav', sr = None)

plt.subplot(3,1,2)
plot_wav(wavefilea)
plt.ylabel('speech')
plt.subplot(3,1,3)
plot_wav(wavefileb)
plt.ylabel('silence')

plt.plot()
plt.show()
#x = np.reshape(np.array(wavefile[0]), (200, -1))
#print x.shape

