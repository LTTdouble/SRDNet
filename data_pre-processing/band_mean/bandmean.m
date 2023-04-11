clear 
clc
close all

dataset = 'CAVE';

%% obtian the original hyperspectral image
src_path =  ['E:\Ëã·¨\MCNet-master\dataset\trains\',dataset,'\4\'];
fileFolder=fullfile(src_path);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name}';
length(fileNames)

for i = 1:length(fileNames)
 name = char(fileNames(i));
 disp(['-----deal with:',num2str(i),'----name:',name]);  
 data_path = [src_path, '/', name];
 load(data_path)
 sizeLR = size(hsi);
%  sizeLR = size(ref);
 band_mean(i,:) = mean(reshape(hsi,[sizeLR(1)*sizeLR(2), sizeLR(3)]));
end

band_mean = mean(band_mean)

