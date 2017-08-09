%All classes
%rng(123)
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);
%train_data(:,16) = [];

reps= 100;
cv_folds = 5;
n_samples=78;
n_features = 345;

tru_indices = {[],[110,149,194,203],[177,185,195,251,317,337],[221,232,251,260],[64,197,300,317]};
%tru_indices = {[],[157,62,24,131,107,35,65,61,83,109,11,59,13],[157,62,24,131,107,35,65,61,83,109,11,59,13],[157,62,24,131,107,35,65,61,83,109,11,59,13],[157,62,24,131,107,35,65,61,83,109,11,59,13]};
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
                 
        tru_svm_pred = multisvm2(tr_dat, tr_lab, ts_dat, tru_indices);
        tru_score = (tru_svm_pred==ts_lab);     
        tru_score = sum(tru_score)/size(tru_score,1); 
        tru_scores_inner(1,i) = tru_score;            
        
        count = count+1;
    end
    tru_scores_final(1,j) = mean(tru_scores_inner)
end   

mean(tru_scores_final)
std(tru_scores_final)
 
