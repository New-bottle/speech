from Tkinter import *
import wave
import math
import matplotlib.pyplot as plt
import numpy as np

#wav1 = '1.wav'
wav1 = 'en_4092_b.wav'
frameSize = 200
overLap = 0
zcr_threshold = 5
ste_threshold = 5
ste_threshold_low = 3

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
	wave_data = np.fromstring(str_data, dtype = np.int8)

	# for the data is stereo, and format is LRLRLR...
	# shape the array to n*2 (-1 means fit the y cordinate)
	wave_data.shape = -1, 2

	# transpose the data
	wave_data = wave_data.T
	# calculate the time bar
	time = np.arange(0, nframes) * (1.0 / framerate)
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
	#frame = enframe(waveData[0], frameSize, frameSize)
	# calculate the short-time-energy
	frame = enframe(waveData[0], frameSize, frameSize)
	energy = sum(frame.T*frame.T) / 1e5
	volume = 10 * np.log(energy)

	# calculate the zero-cross-rate
	tmp1 = enframe(waveData[0][0:len(waveData[0])-1], frameSize, frameSize)
	tmp2 = enframe(waveData[0][1:len(waveData[0])], frameSize, frameSize)
	signs = (tmp1*tmp2) < 0
	diffs = (tmp1-tmp2) > 0
	zcr = sum(signs.T*diffs.T)

	points = []
	status = 0
	last = 0
	for i in range(len(energy)):
		if (status == 0): # 0 : silence
			if energy[i] > ste_threshold or zcr[i] > zcr_threshold:
				print last,i,"sil"
				last = i
				status = 1
				points.append(i)
		elif status == 1: # 1 : speech
			if energy[i] < ste_threshold and zcr[i] < zcr_threshold:
				print last,i,"speech"
				last = i
				status = 0
				points.append(i)

	#draw the wave
	plt.subplot(411)
	plt.plot(time[0:500], waveData[0][0:500], c = 'r')

	time2 = np.arange(0, len(frame))

	plt.subplot(412)
	plt.plot(time2[0:500], energy[0:500], c = "b")
	plt.plot([0,time2[500]], [ste_threshold, ste_threshold], c = 'r')
#	for i in range(len(points)):
#		plt.plot([points[i], points[i]], [0,1e2], c = 'r')
#	energy = short_time_energy(waveData[0], frameSize, overLap)
#	plt.plot(time2, energy, c = "b")
	plt.ylabel("Short-time-energy");

	plt.subplot(413)
	plt.plot(time2[0:500], volume[0:500], c = "b")
	plt.ylabel("volume");

	plt.subplot(414)
	plt.plot(time2[0:500], zcr[0:500], c = "g")
	plt.plot([0,time2[500]], [zcr_threshold, zcr_threshold], c = 'r')
	plt.ylabel("ZCR");
#	zcr = zero_cross_rate(waveData[0], frameSize, overLap)
#	plt.plot(time2, zcr, c = "g")

	plt.plot()
	plt.show()
	return

if __name__ == "__main__":
	main()
