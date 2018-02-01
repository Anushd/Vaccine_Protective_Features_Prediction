%All classes
%rng(100)

%import functional and array data
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

%import labels
lab = data1(:,1);

%arrange data
train_data = [data2(:,2:34),data1(:,2:313)];

%subset features
subset_indices = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/indices_reduced.txt'));
train_data = train_data(:,subset_indices);

%load functional data
train_data1 = train_data(:,1:33);
%load array data
train_data2 = train_data(:,34:293);
%zscore data
train_data1 = zscore(train_data1);
train_data2 = zscore(train_data2);

%define parameters
reps= 100;
cv_folds = 5;
n_samples=78;
num_trees = 100;

%array w/ CV rep scores
scores_final = zeros(1,reps);

%Array with RF-predicted feature importance values for all 345 features,
%for each iteration
features_select1 = zeros(reps*cv_folds,33);
features_select2 = zeros(reps*cv_folds,260);

%array with numerical label predictions
predictions1 = zeros(n_samples,reps);
predictions2 = zeros(n_samples,reps);

count = 1;
for j=1:reps
    %5-fold CV
	indices = crossvalind('Kfold', n_samples, cv_folds);
    %array with CV fold scores
	scores_inner = zeros(1,cv_folds);
    
    for i=1:cv_folds
		%CV train/test indexing
        test = (indices == i); 
        train = ~test;
        ts_dat1 = train_data1(test,:);
        ts_dat2 = train_data2(test,:);
        ts_lab = lab(test,:);
        tr_dat1 = train_data1(train,:);
        tr_dat2 = train_data2(train,:);
        tr_lab = lab(train,:);      
        
		%build models               
        mdl1 = TreeBagger(num_trees,tr_dat1,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        mdl2 = TreeBagger(num_trees,tr_dat2,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        
		%predict labels
        pred1 = predict(mdl1, ts_dat1);
		%convert to double
        pred1 = str2double(pred1); 
        %store predictions (indexing back with CV test indices)
		predictions1(indices==i,j)=pred1;
        
        pred2 = predict(mdl2, ts_dat2);
        pred2 = str2double(pred2); 
        predictions2(indices==i,j)=pred2;
        
        %obtain feature importance predictions and store to matrix
        imp1 = mdl1.OOBPermutedPredictorDeltaError;
        imp2 = mdl2.OOBPermutedPredictorDeltaError;
        %[imp,idx] = sort(imp, 'descend');
        features_select1(count,:) = imp1;
        features_select2(count,:) = imp2;
        
        count = count+1;
    end
    
    pred = 0.2*predictions1 + 0.8*predictions2;   
    pred(pred<=1.8) = 1;
    pred(pred>1.8 & pred<=2.6) = 2;
    pred(pred>2.6 & pred<=3.4) = 3;
    pred(pred>3.4 & pred<=4.2) = 4;
    pred(pred>4.2) = 5;
    
    score = (pred(:,j)==lab);
    score = sum(score)/size(score,1);
    scores_final(1,j) = score
end

pred_a = mean(predictions1,2);
pred_b = mean(predictions2,2);

%color = zeros(78,3);
%color((lab==1),1) = 0;
%color((lab==1),2) = 1;
%color((lab==1),3) = 1;
%color((lab==2),1) = 1;
%color((lab==2),2) = 0;
%color((lab==2),3) = 1;
%color((lab==3),1) = 0;
%color((lab==3),2) = 0;
%color((lab==3),3) = 0;
%color((lab==4),1) = 1;
%color((lab==4),2) = 0;
%color((lab==4),3) = 0;
%color((lab==5),1) = 0;
%color((lab==5),2) = 1;
%color((lab==5),3) = 0;

%scatter(pred_a,pred_b,[],color);

%mean(scores_final)
%std(scores_final)

%features_select_count = zeros(1,n_features);
%for i=1:features_thresh
%    for j=1:reps*cv_folds
%        feature = features_select(j,i);
%        features_select_count(1,feature) = features_select_count(1,feature)+1;
%    end;
%end;  

%for i=1:n_features
%    if features_select_count(1,i)>30
%        i
%    end;
%end;  
