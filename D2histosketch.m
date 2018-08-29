function [ sig_new, val_new] = D2histosketch( a, K, Rand_beta, weight_discri, elem, sig_old, val_old, weight_decay)
% D2histosketch can be used to create or update histosketches
% with 4 inputs to create a sketch from a histogram:
% - a: input histogram (V in the paper)
% - K: sketch length 
% - Rand_beta: -log(\beta) in the paper 
% - weight_discri: discriminative weights of histogram elements
% with 8 inputs to update a sketch from an old sketch and the incoming histogram element:
% - a: updated histogram value V_(t+1) for the incoming histogram element i'
% - K: sketch length
% - Rand_beta: -log(\beta) in the paper 
% - weight_discri: discriminative weights of histogram elements
% - elem: the incoming histogram element i'
% - sig_old: old sketch S(t)
% - val_old: old hash value A(t)
% - weight_decay: pre-computed decay weight exp(-\lambda), where \lambda is the decay factor
% output:
% - sig_new: created/updated sketches
% - val_new: created/updated hash values

switch nargin
    case 4
        a_ind = find(a);
        if weight_discri==0
            temp_mat = Rand_beta(:,a_ind)./repmat(a(:,a_ind),[K,1]);
        else
            temp_mat = Rand_beta(:,a_ind)./repmat(weight_discri(a_ind).*a(:,a_ind),[K,1]);
        end
        [val_new,ind] = min(temp_mat,[],2);
        sig_new = a_ind(ind);
    case 8
        sig_new = sig_old;
        if weight_discri==0
            temp_mat = Rand_beta(:,elem)./repmat(a,[K,1]);
        else
            temp_mat = Rand_beta(:,elem)./repmat(a*weight_discri(elem),[K,1]);
        end
        [val_new, ind] = min([temp_mat,val_old'./weight_decay],[],2);
        sig_new(ind==1) = elem;
        
        
    otherwise
        error('The number of arguements is incorrect !');
end
end

