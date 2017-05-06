function imdb = setupUiuc_Sports(datasetDir, varargin)
% SETUPSCENE67    Setup Flickr Material Dataset
%    This is similar to SETUPCALTECH101(), with modifications to setup
%    the Flickr Material Dataset accroding to the standard
%    evaluation protocols.
%
%    See: SETUPCALTECH101().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.lite = false ;
opts.seed = 1 ;
opts.numTrain = 70 ;
opts.numTest = 60 ;
opts.autoDownload = true ;
opts = vl_argparse(opts, varargin) ;

% Download and unpack
vl_xmkdir(datasetDir) ;
if ~exist(fullfile(datasetDir, 'badminton'))
  error('Scene15 not found in %s', datasetDir) ;
end

imdb = setupGeneric(fullfile(datasetDir), ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', 8, ...
  'seed', opts.seed, 'lite', opts.lite) ;
