function imdb = setupdataset(image_dataset)
imdb.meta.set = cell(1,1);
imdb.meta.set{1} = 'train';
imdb.meta.set{2} = 'test';
imdb.imageDir = fullfile('data',image_dataset);

all_img_dir = dir(fullfile('images',image_dataset));
all_img_dir = all_img_dir(3:end);

imdb.images.id = [1:1:size(all_img_dir,1)];
imdb.images.name = cell(size(all_img_dir,1),1);
for i=1:size(all_img_dir,1)
    imdb.images.name{i} = all_img_dir(i).name;
end
num=1;
classid=1;
imdb.images.class = cell(size(all_img_dir,1),1);
for i=1:size(all_img_dir,1)
    imdb.images.class{i} = all_img_dir(i).name(1:6);
end
while(num<=size(all_img_dir,1))
imdb.meta.classes{classid}=imdb.images.class{num};
it=strcmp(imdb.images.class, imdb.images.class{num});
imdb.meta.num{classid}=sum(it);
num=num+imdb.meta.num{classid};
classid=classid+1;
end
classid=classid-1;


istest=zeros(size(imdb.images.class,1),1);
numoftest=2;
tempnum=0;
for i=1:classid
    if(imdb.meta.num{i}>=10)
        test=randperm(imdb.meta.num{i});
        for n=1:numoftest
            istest(tempnum+test(n))=1;
        end
    end
    tempnum=tempnum+imdb.meta.num{i};
end
imdb.images.istest=istest;


% opts.dataset = image_dataset;
% opts.datasetDir = ['images\'  image_dataset];
% opts.seed = 0;
% opts.lite = false;
% switch opts.dataset
%    case 'scene67', imdb = setupScene67(opts.datasetDir, 'lite', opts.lite) ;
%    case 'caltech101', imdb = setupCaltech256(opts.datasetDir, 'lite', opts.lite, ...
%                                              'variant', 'caltech101', 'seed', opts.seed) ;
%    case 'caltech256', imdb = setupCaltech256(opts.datasetDir, 'lite', opts.lite) ;
%    case 'voc07', imdb = setupVoc(opts.datasetDir, 'lite', opts.lite, 'edition', '2007') ;
%    case 'fmd', imdb = setupFMD(opts.datasetDir, 'lite', opts.lite) ;
%    case 'scene15', imdb = setupScene15(opts.datasetDir, 'lite', opts.lite,'seed', opts.seed) ;
%    case 'uiuc_sports', imdb = setupUiuc_Sports(opts.datasetDir, 'lite', opts.lite,'seed', opts.seed) ;
%    case 'cub200', imdb = setupCUB(opts.datasetDir) ;
%    case 'CUB200-2010', imdb = setupCUB_2010(opts.datasetDir) ;
%    case 'car196', imdb = setupCar(opts.datasetDir) ;
%    case 'sun397', imdb = setupSUN397(opts.datasetDir) ;       
%    otherwise, error('Unknown dataset type.') ;
% end
