from Tkinter import *
import wave
import matplotlib.pyplot as plt
import numpy as np

wav1 = '2.wav'
threshold = 5000
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

len = 10
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
	wave_data, time = read_wave_data(wav1)
	bound_data = find_boundary(time, wave_data)
	#draw the wave
	plt.subplot(311)
	#print bound_data
	plt.scatter(time, wave_data[0], s=5, c=bound_data)
	plt.subplot(312)
	plt.plot(time, wave_data[1], c = "g")
	plt.subplot(313)
	plt.plot(np.fft.fft(wave_data[0]))
	plt.show()

if __name__ == "__main__":
	main()
