x = readwav('en_4092_a.wav');
%x = readwav('1.wav');
y = x';
y = y(1,:);
tmp=vad_cont(y(1:10000));
