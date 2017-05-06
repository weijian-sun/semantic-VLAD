function  pcaData = DimReduce(numChunk, chunk_dir, dim, dim_pca)
feature_num_per_chunk = 500;
whitening = 0;
train_data = cell(1,numChunk) ;
cacheDir = chunk_dir;

for c = 1:numChunk
    chunkPath = fullfile(cacheDir, sprintf('chunk-%0.3d.mat',c));
    load(chunkPath,'data');
    tempdata = data;
    sel = vl_colsubset(1:size(tempdata,2), single(feature_num_per_chunk)) ;
    train_data{c} = tempdata(:,sel);
end

train_data = cat(2,train_data{:});

dim_pca = min(dim_pca, size(train_data,2)-1);

tic;

pcaData.mu = mean(train_data,2);
train_data =  bsxfun(@minus, train_data, pcaData.mu);
[V, ~, D] = princomp(train_data', 'econ');
if whitening
    pcaData.proj = V(:,1:dim_pca)';
else
    pcaData.proj = diag(1./sqrt(D(1:dim_pca) + 1e-5)) * V(:,1:dim_pca)';
end
t = toc;
fprintf('%s: pca  %.2f minutes.\n', mfilename, t/60) ;

clear train_data

% for c = 1:chunk
%     chunkPath = fullfile(cacheDir, sprintf('chunk-%0.3d.mat',c));
%     load(chunkPath,'data');
%     data_ori = data;
%     data = cell(num_block, 1);
%     for b=1:num_block
%         lo = (b-1)*dim + 1;
%         hi = b*dim;
%         data{b} = pcaData.proj{b} * bsxfun(@minus, data_ori(lo:hi,:), pcaData.mu{b});
%     end
%     data = cat(1, data{:});
%     pcaPath = fullfile('feature\evalnorm', sprintf('chunk-%0.3d.mat',c));
%     save(pcaPath, 'data');
% end