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
#	wave_data[0] = wave_data[0] * 1.0 / max(abs(wave_data[0]))
#	wave_data[1] = wave_data[1] * 1.0 / max(abs(wave_data[1]))
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
	#frame = enframe(waveData[0], frameSize, frameSize)
	# calculate the short-time-energy
	frame = enframe(waveData[0], frameSize, frameSize)
	energy = sum(frame.T*frame.T)

	# calculate the zero-cross-rate
	tmp1 = enframe(waveData[0][0:len(waveData[0])-1], frameSize, frameSize)
	tmp2 = enframe(waveData[0][1:len(waveData[0])], frameSize, frameSize)
	signs = (tmp1*tmp2) < 0
	diffs = (tmp1-tmp2) > 0.02
	zcr = sum(signs.T*diffs.T)

	#draw the wave
	plt.subplot(311)
	plt.plot(time, waveData[0], c = 'r')

	time2 = time[np.arange(0, len(time), frameSize)]

	plt.subplot(312)
	plt.plot(time2, energy, c = "b")
#	energy = short_time_energy(waveData[0], frameSize, overLap)
#	plt.plot(time2, energy, c = "b")

	plt.subplot(313)
	plt.plot(time2, zcr, c = "g")
#	zcr = zero_cross_rate(waveData[0], frameSize, overLap)
#	plt.plot(time2, zcr, c = "g")

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
