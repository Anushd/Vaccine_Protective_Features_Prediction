%rng(100)
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);

%train_data(:,16) = [];

reps= 10;
cv_folds = 5;
n_samples=78;
n_features = 345;

tru_indices = [13,156,59,16,40,317];

scores_final = zeros(1,reps);
tru_scores_final = zeros(1,reps);

coefs = zeros(reps*cv_folds,n_features);
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
        
        tru_tr_dat = tr_dat(:,tru_indices);
        tru_ts_dat = ts_dat(:,tru_indices);
        
        mdl = TreeBagger(100,tru_tr_dat,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        pred = predict(mdl, tru_ts_dat);
        pred = str2double(pred);
        score = (pred==ts_lab); 
        score = sum(score)/size(score,1);
        tru_scores_inner(1,i) = score;        
        
        count = count+1;
    end
    tru_scores_final(1,j) = mean(tru_scores_inner)
end   

mean(tru_scores_final)
std(tru_scores_final)

for i=1:n_features
var_count=0;
    for j=1:reps*cv_folds
        if coefs(j,i) ~= 0
            var_count = var_count + 1;
        end;
    end;
    if var_count>200
        i
    end;
end;
