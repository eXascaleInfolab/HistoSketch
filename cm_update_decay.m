function [ cm_table, esti_freq ] = cm_update_decay( data, seed_cm, cm_table, cm_hf_num, cm_hash_len, weight_decay )
% count-min sketch update, and returen the estimated freq at the same time

row_ind = [1:cm_hf_num];
esti_freq = zeros(size(data));
for ii=1:length(data)
    ind = double(randomHash_fast_fixlen( data(ii), seed_cm(1:cm_hf_num), cm_hash_len ));
    idx = (ind-1)*cm_hf_num + row_ind;
    
    cm_table = cm_table*weight_decay;
    cm_table(idx) = cm_table(idx)+1;
    esti_freq(ii) = min(cm_table(idx));
end


end

