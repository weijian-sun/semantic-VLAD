function FileList = setupFileList(image_dataset, imdb, param)

FileList.base_image_dir = ['images\' imdb.imageDir];
FileList.base_feature_dir = ['feature\' sprintf('%s_%s', param.dataset, param.feat.type)];
FileList.base_model_dir = ['model\' imdb.imageDir];
FileList.base_pyramid_dir = ['pyramid\' imdb.imageDir];
mat_file = cell(length(imdb.images.name),1);
for i = 1:length(imdb.images.name)
    if strcmp(image_dataset,'uiuc_sports')
        idx = strfind(imdb.images.name{i},'.');
    else
        idx = strfind(imdb.images.name{i},'.jpg');
    end
    mat_file{i} = [imdb.images.name{i}(1:idx) 'mat'];
end
FileList.imageFileList = fullfile(FileList.base_image_dir,imdb.images.name);
FileList.featureFileList = fullfile(FileList.base_feature_dir,mat_file);      
FileList.modelFileList = fullfile(FileList.base_model_dir,mat_file);
FileList.pyramidFileList = fullfile(FileList.base_pyramid_dir,mat_file);