function [ entro ] = getEntropyWeight( idf_label, num_label, num_label_log, elem)
%GETENTROPYWEIGHT Summary of this function goes here
%   Detailed explanation goes here

switch nargin
    case 3
        idf_label_prob = idf_label./repmat(sum(idf_label,1),[num_label,1]);
        temp = log2(idf_label_prob).*idf_label_prob;
        temp(isnan(temp)==1)=0;
        entro = 1 + sum(temp,1)./num_label_log;
    case 4
        idf_label_elem = idf_label(:,elem);
        idf_label_prob = idf_label_elem./sum(idf_label_elem,1);
        temp = log2(idf_label_prob).*idf_label_prob;
%         temp = -entropy(idf_label_prob);
        temp(isnan(temp)==1)=0;
        entro = 1 + sum(temp,1)./num_label_log;
        
end

end

