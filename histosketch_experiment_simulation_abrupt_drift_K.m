%%  HistoSketch example code: classification of streaming histograms with abrupt drift.
% By varying K_hash (sketch length) and decay_ratio (decay factor), you can reproduce the same results in Figure 6(a) and 7(a) in our paper.
% Note that Figure 6(a) and 7(a) are the average results of 10 repeated trails (average results of accuracy_over_time over 10 runs).

%% Part I: simulating streams with abrupt drift (Please refer to Section V.A. Synthetic Dataset in our paper)
% generate samples
num_distri=2; % two distributions/classes
stream_num_per_class = 1000000; %streaming element per class
% class 1
mu1 = 100;
sigma1 = 20;
distri1 = round(normrnd(mu1,sigma1,[stream_num_per_class,1]));
% class 2
mu2 = 110;
sigma2 = 20;
distri2 = round(normrnd(mu2,sigma2,[stream_num_per_class,1]));


% simulate histogram with abrupt drift
histo_per_distri = 500; % 500 histograms for each distribution
train_per_distri = 0.5*histo_per_distri; % 50% training
test_per_distri = 0.5*histo_per_distri; % 50% testing
element_per_sample = 1000; % 1000 element per histogram

simulated_streams = zeros(histo_per_distri*num_distri*element_per_sample,3); % simulated streaming histograms
counter = 1;
% fast generate simulated data
for ii=1:element_per_sample
    ind_start = (ii-1)*histo_per_distri*num_distri+1;
    ind_end = ii*histo_per_distri*num_distri;
    
    simulated_streams(ind_start:ind_end,1) = [1:histo_per_distri*num_distri]';
    if ii<element_per_sample*0.25 % abrupt drift at 25% of streaming histogram elements
        simulated_streams(ind_start:ind_end,2) = [distri1(counter:counter+histo_per_distri-1);...
            distri2(counter:counter+histo_per_distri-1)];
        simulated_streams(ind_start:ind_end,3) = [ones(histo_per_distri,1);ones(histo_per_distri,1)+1];
    else
        simulated_streams(ind_start:ind_end,2) = [distri1(counter:counter+train_per_distri-1);...
            distri2(counter:counter+test_per_distri-1);...
            distri2(counter+test_per_distri:counter+histo_per_distri-1);...
            distri1(counter+train_per_distri:counter+histo_per_distri-1)];
        simulated_streams(ind_start:ind_end,3) = [ones(train_per_distri,1);...
            ones(test_per_distri,1)+1;...
            ones(train_per_distri,1)+1;...
            ones(test_per_distri,1)];
    end
    counter = counter + histo_per_distri*num_distri;
end

% get histogram index
histo_index = unique(simulated_streams(:,1));
% training histogram index [1:250] and [501:750] 
training_hisot_index = [1:train_per_distri,...
    train_per_distri+test_per_distri+1:train_per_distri*2+test_per_distri]';
% testing histogram index [251:500] and [751:1000] 
testing_histo_index = [train_per_distri+1:train_per_distri+test_per_distri,...
    train_per_distri*2+test_per_distri+1:train_per_distri*2+test_per_distri*2]';


% using histo element index rather than histo element (to improve efficiency in the experiments)
% the element_vocab can be iteratively built over data streams in practice
[element_vocab, ~, element_index] = unique(simulated_streams(:,2));
element_num = length(element_vocab);
simulated_streams(:,2) = element_index; % replace histo element with its index


%% Part II: Streaming testing process
% paramter initialization for histosketch, we precompute element_num for efficiency purpose.
% in practice, there parameters can be iterative expanded
K_hash = 50; % sketch length
decay_ratio = 0.02; % decay factor
weight_decay = exp(-decay_ratio); 
R_k = gamrnd(2,1,[K_hash,element_num]); % r in the paper
C_k = gamrnd(2,1,[K_hash,element_num]); % c in the paper
B_k = rand(K_hash,element_num); % beta in the paper


% initialize hisotsketch for histograms, and its corresponding hash value
sketches = zeros(length(histo_index),K_hash); 
sketches_hashvalue = zeros(length(histo_index),K_hash); 
sketch_first_flag = zeros(length(histo_index),1); % flag for sketch creation
histo_labels = zeros(length(histo_index),1); % labels for histograms (update over streams)

% count min sketch initialization
cm_hf_num = 10; % count min sketch parameter (d in the paper)
cm_hash_len = 50; % count min sketch parameter (g in the paper)
seed_cm = randi(intmax('uint32'),1,cm_hf_num,'uint32'); % random seeds for count min sketch
cm_table = zeros(length(histo_index),cm_hf_num*cm_hash_len); % count-min sketch table: one row for each histogram

% classification configuration
K_classifier = 5; % K for KNN
testing_points = [0.15:0.005:0.75]; % testing points: percentage of histogram element
testing_points_streams = round(testing_points*size(simulated_streams,1)); % actual testing points over streaming histogram
accuracy_over_time = zeros(length(testing_points_streams),1); % classification accuracy

% start the streams and maintain sketches to represent the original histograms
tic;
for ii=1:size(simulated_streams,1)
    if rem(ii,10000) == 0
        disp(ii); % show progress for every 10K elements
    end
    
    temp_histo_ind = simulated_streams(ii,1);
    temp_element_ind = simulated_streams(ii,2);
    histo_labels(temp_histo_ind) = simulated_streams(ii,3);
    
    % update count min sketch table
    [cm_table(temp_histo_ind,:),esti_freq] = cm_update_decay( temp_element_ind, seed_cm, cm_table(temp_histo_ind,:), cm_hf_num, cm_hash_len, weight_decay);
             

    if sketch_first_flag(temp_histo_ind) == 0 % for sketch creation (only once for each histogram)
        temp_histo = zeros(1,element_num);
        temp_histo(temp_element_ind) = 1;
        [sketches(temp_histo_ind,:),sketches_hashvalue(temp_histo_ind,:)] = ...
            histosketch( temp_histo, K_hash, R_k, B_k, C_k);
        sketch_first_flag(temp_histo_ind)=1;
    else % incremental sketch updating
        [sketches(temp_histo_ind,:),sketches_hashvalue(temp_histo_ind,:)] = ...
            histosketch( esti_freq, K_hash, R_k, B_k, C_k, temp_element_ind, sketches(temp_histo_ind,:), sketches_hashvalue(temp_histo_ind,:), weight_decay);
    end
    
    
    % eveluation
    [flag,ind] = ismember(ii,testing_points_streams);
    if flag % evaluating at delai points
        disp('evaluating');
        hit_total = 0;
        for tt=1:length(testing_histo_index)
            hit = KNN_histosketch_fast( sketches(training_hisot_index,:), sketches(testing_histo_index(tt),:), K_classifier, histo_labels(training_hisot_index), histo_labels(testing_histo_index(tt)));
            hit_total = hit_total+hit;
        end
            
        accuracy = hit_total/length(testing_histo_index);
        accuracy_over_time(ind) = accuracy;
        disp(accuracy);
            
    end
end
toc;


% plot the classification over time (number of streaming elements)
% Note that Figure 6(a) and 7(a) in our paper show the average results of accuracy_over_time over 10 runs).
figure;
plot(accuracy_over_time);
xlabel('Streaming histogram elements');
ylabel('Classification accuracy');
set(gca,'Xtick',[0:20:120],'XTickLabel',{'150', '250', '350', '450','550','650','750'});



