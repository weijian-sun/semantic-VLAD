function descr = CalculateFeatureCNN(image_dir, feat_dir, net)
param.minImageSize = 384;
param.feat.scale = 2.^(-0.5:0.5:1.5);
param.feat.type = 'vgg-v19';

feat_dir = [feat_dir(1:end-4),'.mat'];
if ~exist(feat_dir, 'file')
    
    %% prepare image
    im = imread(image_dir) ;
    if size(im,3) < 3
         im = repmat(im, [1 1 3]);
    end
    [height,width,~] = size(im);
    if min(height,width) ~= param.minImageSize    
        im = imresize(im, param.minImageSize/min(height,width), 'bicubic'); 
    end
    im = single(im) ; % note: 255 range

    %% extract features vgg
    [x1, x2, ~] = size(im);
    min_scale = 225.0 / min([x1, x2]);
    descr = cell(1,numel(param.feat.scale));
    for si = 1:numel(param.feat.scale)
      im_ = imresize(im, max( param.feat.scale(si),min_scale)) ;
      [height,width,~] = size(im_);
        if max(height,width) > 1500 
            im_ = imresize(im_, 1500/max(height,width), 'bicubic'); 
        end
      mean_im = imresize(net.meta.normalization.averageImage, [size(im_,1), size(im_,2)]);
      im_ = im_ - mean_im;
%       im_ = gpuArray(im_) ;
      res = vl_simplenn(net, im_) ;
      switch  param.feat.type
        case 'vgg-s'
            feats = gather(res(15).x); 
        case 'vgg-v19'
            feats = gather(res(37).x);   
        case 'vgg-v16'
            feats = gather(res(32).x); 
      end
      feats = reshape(feats, size(feats,1)*size(feats,2), size(feats,3))';
      descr{si} = feats;
    end
    
%     tempuse = descr;
%     load(feat_dir, 'descr');
%     descr{6} =tempuse{4};
%     descr{7} =tempuse{3};
%     descr{8} =tempuse{2};
%     descr{9} =tempuse{1};
    
    sp_make_dir(feat_dir);
    save(feat_dir, 'descr');
    descr = cat(2, descr{:});
else
%     fprintf('load %s\n',feat_dir);
    load(feat_dir, 'descr');
    descr = [cat(2, descr{:})];
end

% if  param.feat.matrix 
%     [u, s, v] = svd(descrs);  
%     descrs = descrs./max(diag(s));
% end
% if  param.feat.norm2
%     descrs = bsxfun(@times, descrs, 1./(sqrt(sum(descrs.^2))+eps)) ;
% end


