from Tkinter import *
import librosa
import wave
import math
import matplotlib.pyplot as plt
import numpy as np

#wav1 = '1.wav'
wav1 = 'en_4092_a.wav'
frameSize = 200
overLap = 0
zcr_threshold = 0.01
ste_threshold = 0.01
#ste_threshold_low = 1

def read_wave_data(file_path):
	wave_data = librosa.load(wav1, sr=None)
	return wave_data

def enframe(waveData, frameSize, stepLen):
	'''
	waveData  : raw data
	frameSize : length of each frame
	stepLen   : inc between frames
	return an array of [frameSize, ceil(len(waveData) / stepLen)]
	'''
	wlen = len(waveData)
	frameNum = int(math.ceil(wlen*1.0/stepLen))
	pad_length = int((frameNum-1)*stepLen+frameSize)
	zeros = np.zeros((pad_length-wlen,))
	pad_signal = np.concatenate((waveData, zeros))

	indices = np.tile(np.arange(0,frameSize),(frameNum,1)) + np.tile(np.arange(0, frameNum*stepLen, stepLen), (frameSize, 1)).T
	indices = np.array(indices, dtype = np.int32)
	frames = pad_signal[indices]
	# To avoid DC bias, we perform mean subtractions on each frame
#	for i in range(frameNum):
#		frames[i] = frames[i] - np.median(frames[i])
	return frames

def main():
	waveData = read_wave_data(wav1)
	waveData = waveData[0]*1.0
	waveData = waveData / max(abs(waveData))
#	wavefft = np.fft.rfft(waveData)
	# calculate the short-time-energy
	frame = enframe(waveData, frameSize, frameSize)
	energy = sum(frame.T*frame.T)
#	volume = 10 * np.log(energy)
	# calculate the zero-cross-rate
	tmp1 = enframe(waveData[0:len(waveData)-1], frameSize, frameSize)
	tmp2 = enframe(waveData[1:len(waveData)], frameSize, frameSize)
	signs = (tmp1*tmp2) < 0
	diffs = (tmp1-tmp2) > 0
	zcr = sum(signs.T*diffs.T)
	points = []
	status = 0
	last = 0
#	ste_threshold = np.average(energy)
#	zcr_threshold = np.average(zcr)
	for i in range(len(energy)):
		if (status == 0): # 0 : silence
			if energy[i] > ste_threshold and zcr[i] > zcr_threshold:
				print last*25,i*25,"sil"
				last = i
				status = 1
				points.append(i)
		elif status == 1: # 1 : speech
			if energy[i] < ste_threshold or zcr[i] < zcr_threshold:
				print last*25,i*25,"speech"
				last = i
				status = 0
				points.append(i)
	if status == 0:
		print last*25, i*25, "sil"
	else:
		print last*25, i*25, "speech"

	#draw the wave
	showLen = 100000
	plt.subplot(311)
	plt.plot(waveData[0:showLen], c = 'r')
	plt.ylabel("Raw wave")
	plt.subplot(312)
	plt.plot(energy[0:showLen // frameSize], c = "b")
	plt.plot([0,showLen // frameSize-1], [ste_threshold, ste_threshold], c = 'r')
	plt.ylabel("Short Time Energy")
	plt.subplot(313)
	plt.plot(zcr[0:showLen // frameSize], c = "g")
	plt.plot([0,showLen // frameSize-1], [zcr_threshold, zcr_threshold], c = 'r')
	plt.ylabel("Zero Cross Rate")
	plt.plot()
	plt.show()
	return

if __name__ == "__main__":
	main()
