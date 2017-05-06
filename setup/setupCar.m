function imdb = setupCar(datasetDir)

splitDir = fullfile(datasetDir,'devkit');
datasetDir = fullfile(datasetDir,'images');

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;

% read class names
imdb.meta.sets = {'train', 'val', 'test'} ;
load(fullfile(splitDir, 'cars_meta.mat')) ;
imdb.meta.classes = class_names;

% read images info
load(fullfile(splitDir, 'cars_train_annos.mat')) ;
num_train = numel(annotations);
imdb.images.id = 1:num_train;
train_name = cell(1, num_train);
for i = 1:num_train
    train_name{i} = [ '0' annotations(i).fname];
end

load(fullfile(splitDir, 'cars_test_annos.mat')) ;
num_test = numel(annotations);
test_name = cell(1, num_test);
for i = 1:num_test
    test_name{i} = [ '0' annotations(i).fname];
end
imdb.images.name = [train_name, test_name];
imdb.images.id = 1:numel(imdb.images.name);

% read train labels
labelsPath = fullfile(splitDir, 'train_perfect_preds.txt') ;
[labels] = textread(labelsPath, '%f') ;
imdb.images.class = labels' ;
imdb.images.set = zeros(1, numel(imdb.images.name));
imdb.images.set(1:num_train) = 1;
imdb.images.set(num_train+1:num_train+num_test) = 3;

imdb.imageDir = datasetDir;



