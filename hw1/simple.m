clear all
[signal,fs,bit] = wavread('1.wav');
subplot(2,1,1)
plot(signal)
title('语音信号波形')
framelength = 150;
framenumber = fix(length(signal)/framelength);
for i = 1:framenumber;
	framesignal = signal((i-1)*framelength+1:i*framelength);
	Z(i)=0;
	for j=2:framelength-1;
		Z(i)=Z(i)+abs(sign(framesignal(j))-sign(framesignal(j-1)));
	end
end
subplot(2,1,2)
plot(Z)
title('短时平均过零率')