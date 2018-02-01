%import functional and array data
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));
%import labels
lab = data1(:,1);
%arrange data and calculate z-scores
train_data = [data2(:,2:34),data1(:,2:313)];
train_data = zscore(train_data);

%subset features
subset_indices = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/indices_reduced.txt'));
train_data = train_data(:,subset_indices);

%define constants
reps= 100; %number of iterations
cv_folds = 5;
n_samples=78;
n_features = 293;
num_trees = 100;

%Array to store the classification score from each repition
scores_final = zeros(1,reps);

%Array with RF-predicted feature importance values for all 345 features,
%for each iteration
features_select = zeros(reps*cv_folds,n_features);

count = 1;
for j=1:reps
    indices = crossvalind('Kfold', n_samples, cv_folds);
    
    %stores classfication accuracy of each test fold; resets every
    %iteration
    scores_inner = zeros(1,cv_folds);
    
    for i=1:cv_folds
        %partition data into test and training folds
        test = (indices == i); 
        train = ~test;
        ts_dat = train_data(test,:);
        ts_lab = lab(test,:);
        tr_dat = train_data(train,:);
        tr_lab = lab(train,:);      
        
        %build RF model
        mdl = TreeBagger(num_trees,tr_dat,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        
        %predict labels
        pred = predict(mdl, ts_dat);
        pred = str2double(pred);
        
        %check if predicted labels match actual
        score = (pred==ts_lab); 
        %calculate score
        score = sum(score)/size(score,1);
        %store classification accuracy to scores_inner matrix
        scores_inner(1,i) = score;
        
        %obtain feature importance predictions and store to matrix
        imp = mdl.OOBPermutedPredictorDeltaError;
        %[imp,idx] = sort(imp, 'descend');
        features_select(count,:) = imp; 
              
        count = count+1;
    end
    
    %store score (mean of classification accuracies) to scores_final
    scores_final(1,j) = mean(scores_inner)
end   

mean(scores_final)
std(scores_final)
