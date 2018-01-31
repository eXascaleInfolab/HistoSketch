function [ sig_new, val_new] = histosketch( a, K, R_k, B_k, C_k, elem, sig_old, val_old, weight_decay)
% histosketch can be used to create or update histosketches
% with 5 inputs to create a sketch from a histogram:
% - a: input histogram (V in the paper)
% - K: sketch length 
% - R_k, B_k, C_k: parameters r, \beta, c in the paper 
% with 9 inputs to update a sketch from an old sketch and the incoming histogram element:
% - a: updated histogram value V_(t+1) for the incoming histogram element i'
% - K: sketch length
% - R_k, B_k, C_k: parameters r, \beta, c in the paper 
% - elem: the incoming histogram element i'
% - sig_old: old sketch S(t)
% - val_old: old hash value A(t)
% - weight_decay: pre-computed decay weight exp(-\lambda), where \lambda is the decay factor
% output:
% - sig_new: created/updated sketches
% - val_new: created/updated hash values


% a = train_data_mat_freq_decay(ii,:);

switch nargin
    case 5
        a_ind = find(a);
        
        Y_ka = exp(repmat(log(a(a_ind)),[K,1])-R_k(:,a_ind).*B_k(:,a_ind));
        
        A_ka = C_k(:,a_ind)./(Y_ka.*exp(R_k(:,a_ind)));
        
        [val_new, ind] = min(A_ka,[],2);
        
        val_new = val_new';
        if length(a_ind)==1
            sig_new = a_ind(ind);
        else
            sig_new = a_ind(ind)';
        end
    case 9
        sig_new = sig_old;
        
        Y_ka = exp(repmat(log(a),[K,1])-R_k(:,elem).*B_k(:,elem));

        A_ka = C_k(:,elem)./(Y_ka.*exp(R_k(:,elem)));
        
        [val_new, ind] = min([A_ka,val_old'./weight_decay],[],2);
        sig_new(ind==1) = elem;
        
    otherwise
        error('The number of arguements is incorrect !');
end
end

