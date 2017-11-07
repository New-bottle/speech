x = readwav('1.wav');
x([1:10000],:)=[];
y = x';
y = y(1,:);
tmp=vad_cont(y)
