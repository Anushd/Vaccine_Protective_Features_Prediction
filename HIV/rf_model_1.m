%All classes
rng(100)
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);

reps= 100;
cv_folds = 5;
n_samples=78;
n_features = 345;
num_trees = 100;
%features_thresh = 20;

%tru_indices = {[],[110,149,194,203],[177,185,195,251,317,337],[221,232,251,260],[64,197,300,317]};
scores_final = zeros(1,reps);
tru_scores_final = zeros(1,reps);

features_select = zeros(reps*cv_folds,n_features);

count = 1;
for j=1:reps
    indices = crossvalind('Kfold', n_samples, cv_folds);
    scores_inner = zeros(1,cv_folds);
    tru_scores_inner = zeros(1,cv_folds);
    
    for i=1:cv_folds
        test = (indices == i); 
        train = ~test;
        ts_dat = train_data(test,:);
        ts_lab = lab(test,:);
        tr_dat = train_data(train,:);
        tr_lab = lab(train,:);      
                       
        mdl = TreeBagger(num_trees,tr_dat,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
 
        pred = predict(mdl, ts_dat);
        pred = str2double(pred);
        score = (pred==ts_lab); 
        score = sum(score)/size(score,1);
        scores_inner(1,i) = score;
        
        imp = mdl.OOBPermutedPredictorDeltaError;
        %[imp,idx] = sort(imp, 'descend');
        features_select(count,:) = imp; 
              
        count = count+1;
    end
    scores_final(1,j) = mean(scores_inner)
end   

mean(scores_final)
std(scores_final)

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
