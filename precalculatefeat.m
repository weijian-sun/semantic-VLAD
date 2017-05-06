function precalculatefeat(images, feat, varargin)
%% Step 0: obtain sample image descriptors
numImages = numel(images) ;
% gpuDevice(1);
switch 'vgg-v19'
    case 'vgg-f'
        net = load('imagenet-vgg-f.mat') ;
        net.layers = net.layers(1:14);
    case 'vgg-m'
        net = load('imagenet-vgg-m.mat') ;
        net.layers = net.layers(1:14);
    case 'vgg-s'
        net = load('imagenet-vgg-s.mat') ;
        net.layers = net.layers(1:14);
    case 'vgg-v16'
        net = load('imagenet-vgg-verydeep-16.mat') ;
        net.layers = net.layers(1:31);
    case 'vgg-v19'
        net = load('imagenet-vgg-verydeep-19.mat') ;
        net.layers = net.layers(1:36);
end
% net = vl_simplenn_move(net, 'gpu') ;

for i = 1:numImages
    fprintf('load/calculate %d\n',i);
    CalculateFeatureCNN(images{i}, feat{i}, net);
end
