function imdb = setupBirds(datasetDir, varargin)
% [S. Lazebnik, C. Schmid, and J. Ponce. A Maximum Entropy Framework for 
% Part-Based Texture and Object Recognition.ICCV,2005.] 
% 6 bird classes, 100 images per class. We randomly select 50 images for
% training and the rests for test, which is different with the original
% paper.

opts.lite = false ;
opts.seed = 1 ;
opts.numTrain = 50 ;
opts.numTest = 50 ;
opts.autoDownload = true ;
opts = vl_argparse(opts, varargin) ;

vl_xmkdir(datasetDir) ;
imdb = setupGeneric(fullfile(datasetDir,'image'), ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', 10, ...
  'seed', opts.seed, 'lite', opts.lite) ;
