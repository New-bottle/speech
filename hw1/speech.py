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
zcr_threshold = 8
ste_threshold = 8
ste_threshold_low = 3

def read_wave_data(file_path):
	wave_data = librosa.load(wav1, sr=None)
	time = np.arange(0, wave_data[0].shape[0])
	return wave_data, time

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
	for i in range(frameNum):
		frames[i] = frames[i] - np.median(frames[i])
	return frames

def short_time_energy(waveData, frameSize, overLap):
	wlen = len(waveData)
	step = frameSize - overLap
	frameNum = int(math.ceil(wlen*1.0/step))
	energy = np.zeros((frameNum,1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i*step, min(i*step+frameSize, wlen))]
		curFrame = curFrame - np.median(curFrame) # zero-justified
		energy[i] = np.sum(curFrame*curFrame)
	return energy

def zero_cross_rate(waveData, frameSize, overLap):
	wlen = len(waveData)
	step = frameSize - overLap
	frameNum = int(math.ceil(wlen*1.0/step))
	crossRate = np.zeros((frameNum,1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i*step, min(i*step+frameSize, wlen-1))]
		nxtFrame = waveData[np.arange(i*step+1, min(i*step+frameSize+1, wlen))]
		crossRate[i] = sum(abs(np.sign(curFrame) - np.sign(nxtFrame))) / 2.0 / frameSize
	return crossRate

def main():
	waveData, time = read_wave_data(wav1)
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
	plt.plot(time[0:showLen], waveData[0:showLen], c = 'r')
	time2 = np.arange(0, len(frame))
	plt.subplot(312)
	plt.plot(time2[0:showLen // frameSize], energy[0:showLen // frameSize], c = "b")
	plt.plot([0,time2[showLen // frameSize-1]], [ste_threshold, ste_threshold], c = 'r')
#	for i in range(len(points)):
#		plt.plot([points[i], points[i]], [0,1e2], c = 'r')
#	energy = short_time_energy(waveData, frameSize, overLap)
#	plt.plot(time2, energy, c = "b")
	plt.ylabel("Short-time-energy");
#	plt.subplot(313)
#	plt.plot(wavefft)
#	plt.plot(time2[0:showLen], volume[0:showLen], c = "b")
#	plt.ylabel("volume");
	plt.subplot(313)
	plt.plot(time2[0:showLen // frameSize], zcr[0:showLen // frameSize], c = "g")
	plt.plot([0,time2[showLen // frameSize-1]], [zcr_threshold, zcr_threshold], c = 'r')
	plt.ylabel("ZCR");
#	zcr = zero_cross_rate(waveData, frameSize, overLap)
#	plt.plot(time2, zcr, c = "g")
	plt.plot()
	plt.show()
	return

if __name__ == "__main__":
	main()
