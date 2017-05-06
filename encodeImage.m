function encodeImage(encoder, im, feat, varargin)
% ENCODEIMAGE   Apply an encoder to an image
%   DESCRS = ENCODEIMAGE(ENCODER, IM) applies the ENCODER
%   to image IM, returning a corresponding code vector PSI.
%
%   IM can be an image, the path to an image, or a cell array of
%   the same, to operate on multiple images.
%
%   ENCODEIMAGE(ENCODER, IM, CACHE) utilizes the specified CACHE
%   directory to store encodings for the given images. The cache
%   is used only if the images are specified as file names.
%
%   See also: TRAINENCODER().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
opts.usepreprocessfeature=false;
opts.codedimention = 0;
opts.step = -1;
opts.factor = 0;
opts.nettype = 'imagenet-vgg-m.mat';
opts.type = 'bovw' ;
opts.numWords = [] ;
opts.seed = 1 ;
opts.numPcaDimensions = +inf ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.numSamplesPerWord = [] ;
opts.renormalize = false ;
opts.layouts = {'1x1'} ;
opts.geometricExtension = 'none' ;
opts.subdivisions = zeros(4,0) ;
opts.readImageFn = @readImage2 ;
opts.extractorFn = @getDenseCnn ;
opts.lite = false ;

opts.cacheDir = [] ;
opts.cacheChunkSize = 512 ;
opts = vl_argparse(opts,varargin) ;

if ~iscell(im), im = {im} ; end

% break the computation into cached chunks
startTime = tic ;
descrs = cell(1, numel(im)) ;
numChunks = ceil(numel(im) / opts.cacheChunkSize) ;

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


for c = 1:numChunks
    n  = min(opts.cacheChunkSize, numel(im) - (c-1)*opts.cacheChunkSize) ;
    chunkPath = fullfile(opts.cacheDir, sprintf('chunk-%03d.mat',c)) ;
    if ~isempty(opts.cacheDir) && exist(chunkPath)
        continue
    else
        range = (c-1)*opts.cacheChunkSize + (1:n) ;
        fprintf('%s: processing a chunk of %d images (%3d of %3d, %5.1fs to go)\n', ...
            mfilename, numel(range), ...
            c, numChunks, toc(startTime) / (c - 1) * (numChunks - c + 1)) ;
        data = processChunk(encoder, im(range), feat(range), net) ;
        if ~isempty(opts.cacheDir)
            save(chunkPath, 'data', '-v7.3') ;
        end
    end
    %descrs{c} = data ;
    clear data ;
end
% for c = 1:numChunks
%     fprintf('%s: loading descriptors from %s\n', mfilename, chunkPath) ;
%     chunkPath = fullfile(opts.cacheDir, sprintf('chunk-%03d.mat',c)) ;
%     load(chunkPath, 'data') ;
%     descrs{c}=data;
% end
% descrs = cat(2,descrs{:}) ;

% --------------------------------------------------------------------
function psi = processChunk(encoder, im, feat, net)
% --------------------------------------------------------------------
psi = cell(1,numel(im)) ;
parfor i = 1:numel(im)
%     fprintf('process %d\n',i);
    psi{i} = encodeOne(encoder, im{i}, feat{i}, net) ;
end
psi = cat(2, psi{:}) ;

% --------------------------------------------------------------------
function psi = encodeOne(encoder, im, feat, net)
% --------------------------------------------------------------------
features = CalculateFeatureCNN(im, feat, net);
descrs = encoder.projection * bsxfun(@minus, ...
    features, encoder.projectionCenter) ;
if encoder.renormalize
    descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

switch encoder.type
    case {'crosemodel'}
        z = tempcacu(descrs ,encoder.mean, encoder.covariance, encoder.weight);
    case {'fv','cnn-fv'}
        %z = fisher_encode(descrs, encoder.means, encoder.covariances, encoder.priors);
        z = vl_fisher(descrs, ...
            encoder.means, ...
            encoder.covariances, ...
            encoder.priors, ...
            'Improved') ;
    case 'vlad'
        [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
            descrs, ...
            'MaxComparisons', 15) ;
        assign = zeros(encoder.numWords, numel(words), 'single') ;
        assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
        z = vl_vlad(descrs, ...
            encoder.words, ...
            assign, ...
            'SquareRoot', ...
            'NormalizeComponents') ;
end
z = z / max(sqrt(sum(z.^2)), 1e-12) ;
psi = z;

