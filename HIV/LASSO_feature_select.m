%All classes
%rng(123)
data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);

reps= 10;
cv_folds = 5;
n_samples=78;
n_features = 345;

classes = unique(lab);
num_classes = length(classes);

mse = [0,0,0,0,0];
thresh = [25,25,25,25,25];

coefs_1 = zeros(reps*cv_folds,n_features);
coefs_2 = zeros(reps*cv_folds,n_features);
coefs_3 = zeros(reps*cv_folds,n_features);
coefs_4 = zeros(reps*cv_folds,n_features);
coefs_5 = zeros(reps*cv_folds,n_features);
coefs = {coefs_1, coefs_2, coefs_3, coefs_4, coefs_5};
count = 1;

for j=1:reps
    indices = crossvalind('Kfold', n_samples, cv_folds);
    scores_inner = zeros(1,cv_folds);
    tru_scores_inner = zeros(1,cv_folds);
    
    for i=1:cv_folds
        test = (indices == i); 
        train = ~test;
        tr_dat = train_data(train,:);
        tr_lab = lab(train,:);
        
        for k=1:num_classes
            tr_lab_tmp = (tr_lab==k);
            [B, stats] = lasso(tr_dat,tr_lab_tmp,'CV',5, 'Standardize', false);
            betas = B(:,stats.IndexMinMSE-mse(k))';                          
            coefs{k}(count,:) =  betas;                                  
        end     
        count = count+1;
    end
end

for k=1:num_classes
    disp('Class')
    disp(k)
    for i=1:n_features
    var_count=0;
        for j=1:reps*cv_folds
            if coefs{k}(j,i) ~= 0
                var_count = var_count + 1;
            end;
        end;
        if var_count>thresh(k)
            i
        end;
    end;  
end;
