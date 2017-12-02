from Tkinter import *
import wave
import math
import matplotlib.pyplot as plt
import numpy as np

wav1 = '1.wav'
frameSize = 256
overLap = 128
def read_wave_data(file_path):
	# open a wave file, and return a Wave_read object
	f = wave.open(file_path, 'rb')
	# read the wave's format information, and return a tuple
	params = f.getparams()
	# get the info
	nchannels, sampwidth, framerate, nframes = params[:4]
	# read and return nframes of audion, as a strign of bytes
	str_data = f.readframes(nframes)
	# close the stream
	f.close()

	#turn the wave's data to array
	wave_data = np.fromstring(str_data, dtype = np.short)

	# for the data is stereo, and format is LRLRLR...
	# shape the array to n*2 (-1 means fit the y cordinate)
	wave_data.shape = -1, 2

	# transpose the data
	wave_data = wave_data.T
	# calculate the time bar
	time = np.arange(0, nframes) * (1.0 / framerate)
	return wave_data, time

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
	wlen = len(waveData) - 1
	step = frameSize - overLap
	frameNum = int(math.ceil(wlen*1.0/step))
	crossRate = np.zeros((frameNum,1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i*step, min(i*step+frameSize, wlen))]
		curFrame = curFrame - np.median(curFrame) # zero-justified
		for j in range(len(curFrame) - 1):
			if (curFrame[j] * curFrame[j + 1] <= 0):
				crossRate[i] += 1
	return crossRate

def find_boundary(time, wave_data):
	bound_data = ['k'] * time.shape[0]
	for i in range(time.shape[0] - len):
		flag = True
		count = 0
		for j in range(len):
			if abs(wave_data[0][i+j]) < threshold:
				count = count + 1
		if count < len / 3:
			bound_data[i] = 'k'
		else:
			bound_data[i] = 'r'
	return bound_data

def main():
	waveData, time = read_wave_data(wav1)
	#draw the wave
	plt.subplot(311)
	plt.plot(time, waveData[0], c = 'r')

	plt.subplot(312)
	energy = short_time_energy(waveData[0], frameSize, overLap)
	time2 = time[0:len(energy)]
	plt.plot(time2, energy, c = "b")
	print energy

	plt.subplot(313)
	zcr = zero_cross_rate(waveData[0], frameSize, overLap)
	plt.plot(time2, zcr, c = "g")
	plt.plot()
	plt.show()

if __name__ == "__main__":
	main()
