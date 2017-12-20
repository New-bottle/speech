# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2017-10-23 17:19:05
# @Last Modified by:   richman
# @Last Modified time: 2017-11-09 16:30:57

import argparse
import librosa
import numpy as np
parser = argparse.ArgumentParser()
""" Arguments: inwav, idxlist,outwav """
parser.add_argument('inwav',help="Input wave file", type=str)
parser.add_argument('idxlist',help="Index list, specifying all the cut indexes e.g.,: 0 100 sil", type=argparse.FileType('r'))
parser.add_argument('outwav',help="Output wave file. Will be generated. Default: %{default}s",type=str, default="cutted.wav")
parser.add_argument('-sl','--speechlabel', type=str, help="Speechlabel, default: %(default)s",default="sil")

args = parser.parse_args()

origaudio, sr = librosa.load(args.inwav, sr=None)

resaudio = []
for line in args.idxlist:
	begin,end, classid = line.strip().split()
	if classid == args.speechlabel:
		scale = sr // 1000
		begin,end = int(begin) * scale, int(end) *scale
		resaudio.extend(origaudio[begin:end])

resaudio = np.array(resaudio).flatten()
librosa.output.write_wav(args.outwav,resaudio, sr)
