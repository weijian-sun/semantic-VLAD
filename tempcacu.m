function z = tempcacu(descrs, mean, covariance, weight)
% descrs = bsxfun(@times, descrs, 1./(sqrt(sum(descrs.^2))+eps)) ;
signdescrs = bsxfun(@times, (descrs), 1./((sum((descrs)))+eps)) ;
[n,m] = size(descrs);
z = cell(512,1);

for i = 1:512
    id = find(descrs(weight(i),:)>0);
    descrstemp = descrs(:,id);
    temp = signdescrs(weight(i),id);
%     z{i} = descrstemp*temp';
    z{i} = bsxfun(@minus, descrstemp, mean{weight(i)})*temp';
    z{i} = norm(z{i});
end
z = cat(1, z{:});
%z = z./max(sqrt(sum(z.^2)), eps(4));
% z = norm(z);
end

function z = norm(z)
z = sign(z).*(abs(z)).^0.85;
z = z./max(sqrt(sum(z.^2)), eps(4));
end