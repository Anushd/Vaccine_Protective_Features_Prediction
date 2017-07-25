%Class A: elite controllers; Class B: cart treated & chronic untreated
rng(100)
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);
group1_indices = find(lab==2 | lab==4);
group34_indices = find(lab==3 | lab==5);
lab(group34_indices) = 0;
lab(~group34_indices) = 1;
lab = lab(sort([group1_indices; group34_indices]));

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);
train_data = train_data(sort([group1_indices; group34_indices]),:);
%train_data(:,16) = [];

reps= 10;
cv_folds = 5;

lab_size = size(lab);
n_samples=lab_size(1);
n_features = 345;

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
        [B, stats] = lasso(tr_dat,tr_lab, 'CV', 5);
        betas = B(:,stats.IndexMinMSE)';
        coefs(count,:) =  betas;        
       
        count = count+1;
    end

end   

for i=1:n_features
var_count=0;
    for j=1:reps*cv_folds
        if coefs(j,i) ~= 0
            var_count = var_count+1;
        end;
    end;
    if var_count>30
        i
    end;
end;
