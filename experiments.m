function experiments()
% EXPERIMENTS   Run image classification experiments
%    The experimens download a number of benchmark datasets in the
%    'data/' subfolder. Make surce that there are several GBs of
%    space available.
%
%    By default, experiments run with a lite option turned on. This
%    quickly runs all of them on tiny subsets of the actual data.
%    This is used only for testing; to run the actual experiments,
%    set the lite variable to false.
%
%    Running all the experiments is a slow process. Using parallel
%    MATLAB and several cores/machiens is suggested.

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

lite = false ;
clear ex ;

% fprintf('download vgg-vd19');
% urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', ...
%          'imagenet-vgg-verydeep-19.mat') ;

ex(1).prefix = 'cnnvd37-fv-146';
ex(1).trainOpts = {'C', 10} ;
% ex(1).datasets = {'cud_2010'} ;
% ex(1).datasets = {'scene67'} ;
ex(1).datasets = {'fmd'};
ex(1).seed = 1;
ex(1).usepreprocessfeature = true;
ex(1).imagedateDir = 'feat';
ex(1).opts = {
  'type', 'crosemodel', ...
  'numWords', 256, ...
  'layouts', {'1x1'}, ...
  'geometricExtension', 'none', ...
  'numPcaDimensions',+inf, ...
%    'extractorFn', @(x) getDenseCnn(x, net, 'netlevel', 37, 'scales', 2.^(-0.5:.5:2))
  }; 

if lite, tag = 'lite' ;
else, tag = 'ex' ; end

%for i=1:numel(ex)
for i=1
  for j=1:numel(ex(i).datasets)
    dataset = ex(i).datasets{j} ;
    if ~isfield(ex(i), 'trainOpts') || ~iscell(ex(i).trainOpts)
      ex(i).trainOpts = {} ;
    end
    traintest(...
      'imagedateDir', ex(i).imagedateDir,...
      'prefix', [tag '-' dataset '-' ex(i).prefix], ...
      'dataset', char(dataset), ...
      'datasetDir', fullfile('data', dataset), ...
      'lite', lite, ...
      ex(i).trainOpts{:}, ...
      'encoderParams', ex(i).opts) ;
  end
end
