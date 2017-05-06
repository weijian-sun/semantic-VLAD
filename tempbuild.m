function [means, covariance, p] = tempbuild(descrs)
% descrs = bsxfun(@times, descrs, 1./(sqrt(sum(descrs.^2))+eps)) ;
for i=1:512
    id = find(descrs(i,:)>0);

    means{i} = mean(descrs(:,id),2);
%     means{i}(find(means{i}==0))=1;
%     means{i} = mean(temp, 2);
%     covariance{i} = sum(bsxfun(@minus, temp, means{i}).^2,2)/size(temp,2);
    covariance{i} = var(descrs(:,id)')';
%     mean{i} = descrs*temp'/sum(temp,2);
end
weight = sum(descrs,2);
% weight = var(descrs')';
weight = weight./max(sum(weight), eps(4));
[i,p] = sort(weight, 1, 'descend');

end