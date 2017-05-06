function imdb = setupCUB(datasetDir)

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;

% read class names
imdb.meta.sets = {'train', 'val', 'test'} ;
classPath = fullfile(datasetDir, 'classes.txt') ;
[index,className] = textread(classPath, '%f %s') ;
imdb.meta.classes = className;

% read images info
imagesPath = fullfile(datasetDir, 'images.txt') ;
[index,imageName] = textread(imagesPath, '%f %s') ;
imdb.images.id = index' ;
imdb.images.name = imageName' ;

labelsPath = fullfile(datasetDir, 'image_class_labels.txt') ;
[index,labels] = textread(labelsPath, '%f %f') ;
imdb.images.class = labels' ;

setPath = fullfile(datasetDir, 'train_test_split.txt') ;
[index,sets] = textread(setPath, '%f %f') ;  % 1-is training image 
sets = ~(sets);
imdb.images.set = double( sets.*2 +1)';

imdb.imageDir = fullfile(datasetDir, 'images') ;
% imdb.imageDir = fullfile(datasetDir, 'bb') ;


