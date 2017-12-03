from Tkinter import *
import wave
import math
import matplotlib.pyplot as plt
import numpy as np

wav1 = '1.wav'
#wav1 = 'en_4092_a.wav'
frameSize = 200
overLap = 0
zcr_threshold = 45
ste_threshold = 4000

def read_wave_data(file_path):
	# open a wave file, and return a Wave_read object
	f = wave.open(file_path, 'rb')
	# read the wave's format information, and return a tuple
	params = f.getparams()
	# get the info
	nchannels, sampwidth, framerate, nframes = params[:4]
	print nchannels, sampwidth, framerate, nframes
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
	print time
	return wave_data, time

def short_time_energy(waveData, frameSize, overLap):
	wlen = len(waveData)
	step = frameSize - overLap
	frameNum = int(math.ceil(wlen*1.0/step))
	energy = np.zeros((frameNum,1))
	for i in range(frameNum):
		curFrame = waveData[np.arange(i*step, min(i*step+frameSize, wlen))]
#		curFrame = curFrame - np.median(curFrame) # zero-justified
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
#		if i == 2:
#			print curFrame
#		curFrame = curFrame - np.median(curFrame) # zero-justified
#		if i == 2:
#			print curFrame
#		for j in range(len(curFrame) - 1):
#			if (int(curFrame[j]) * int(curFrame[j + 1]) <= 0):
#				crossRate[i] += 1
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
	print waveData.shape
	#draw the wave
	plt.subplot(311)
	plt.plot(time, waveData[0], c = 'r')

	plt.subplot(312)
	energy = short_time_energy(waveData[0], frameSize, overLap)
#	time2 = time[0:len(energy)]
	time2 = time[np.arange(0, len(time), frameSize)]
	plt.plot(time2, energy, c = "b")

	plt.subplot(313)
	zcr = zero_cross_rate(waveData[0], frameSize, overLap)
	print zcr
	plt.plot(time2, zcr, c = "g")
	plt.plot()
	plt.show()

	points = []
	status = 0
	for i in range(len(energy)):
		if (status == 0 or status == 1):
			if energy[i] > ste_threshold:
				status = 2
	return

if __name__ == "__main__":
	main()
