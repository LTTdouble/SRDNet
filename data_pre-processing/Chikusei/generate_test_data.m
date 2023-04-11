% fileFolder=fullfile('.\test\');
fileFolder=fullfile('E:\Ëã·¨\mcodes\test\');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
factor = 0.125;
img_size = 512;
bands = 128;
hr = zeros(numel(fileNames),img_size,img_size,bands);
lr = zeros(numel(fileNames),img_size*factor,img_size*factor,bands);
cd test;
for i = 1:numel(fileNames)
    load(fileNames{i},'test');
    img_ms = single(imresize(test, factor));
%     hr(i,:,:,:) = test;
%     lr(i,:,:,:) = img_ms;
     hr = test;
     lr = img_ms;
     lr = permute(lr, [3 1 2]);
     hr = permute(hr, [3 1 2]);

% cd ..;
     hr = single(hr);
     lr = single(lr);
    % save('.\dataset\Chikusei_x4\Chikusei_test.mat','hr','lr');
    % file_path = strcat('.\dataset\tests ','mat');
     file_path = strcat('E:\Ëã·¨\mcodes\dataset\tests\Chikusei_test_', int2str(i), '.mat');
     save(file_path,'hr','lr','-v6');
end