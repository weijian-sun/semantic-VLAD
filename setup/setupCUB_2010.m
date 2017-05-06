function imdb = setupCUB_2010(datasetDir)

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;

imdb.meta.sets = {'train', 'val', 'test'} ;
splitPath = fullfile(datasetDir, 'splits.mat') ;
load(splitPath);
imdb.meta.classes = splits.classes;
imdb.images.name = [splits.train_files, splits.test_files] ;
imdb.images.class = [splits.train_labels, splits.test_labels] ;
imdb.images.set = [ones(1,length(splits.train_files)), 3 * ones(1,length(splits.test_files))];
imdb.images.id = 1:length(imdb.images.name);

imdb.imageDir = fullfile(datasetDir, 'images') ;

imdb.featDir = fullfile(datasetDir,'feat');
imdb.setupdata = @(x)setupFMD(x);