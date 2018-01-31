function [ flag, tab ] = KNN_histosketch_fast( training_data, test, N, cate_train, cate_test)
% Customized KNN for sketches with improved runtime efficiency

temp = bsxfun(@minus,training_data,test);

dist = sum(temp==0,2)./size(training_data,2);
[~,I] = sort(dist, 'descend');

flag = 0;

[temp_inds, ~, n] = unique(cate_train(I(1:N)));
tab = tabulate(n);
[~, ind] = max(tab(:,2));
label = temp_inds(ind);
if cate_test == label
    flag = 1;
end

    

end
