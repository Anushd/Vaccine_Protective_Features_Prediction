data = table2array(readtable('/Users/anush/documents/projects/ragon/data/Post_pass1_parsed.txt'));
train_data = data(:,2:81);
%train_data = train_data(train_data(:,81)==1,:);
%train_data = train_data(:,1:80);
lab = data(:,83);
train_data = zscore(train_data);

reps= 100;
cv_folds = 5;
n_samples=46;

tru_indices = [55,58,75];

scores_final = zeros(1,reps);
tru_scores_final = zeros(1,reps);

coefs = zeros(reps*cv_folds,80);
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
        
        features_index = find(betas);
        sel_tr_dat = tr_dat(:,features_index);
        sel_ts_dat = ts_dat(:,features_index);
        
        tru_tr_dat = tr_dat(:,tru_indices);
        tru_ts_dat = ts_dat(:,tru_indices);
        
        if size(sel_tr_dat) ~= 0   
            svm = fitcsvm(sel_tr_dat, tr_lab, 'BoxConstraint', 0.1);
            optimise = hyperparameters('fitcsvm',sel_tr_dat,tr_lab);
            pred = predict(svm, sel_ts_dat);
            score = (pred==ts_lab); 
            score = sum(score)/size(score,1);
            scores_inner(1,i) = score;
        
            tru_svm = fitcsvm(tru_tr_dat, tr_lab, 'BoxConstraint', 0.1);
            tru_pred = predict(tru_svm, tru_ts_dat);
            tru_score = (tru_pred==ts_lab);     
            tru_score = sum(tru_score)/size(tru_score,1); 
            tru_scores_inner(1,i) = tru_score;        
        end    
        
        count = count+1;
    end
    scores_final(1,j) = mean(scores_inner)
    tru_scores_final(1,j) = mean(tru_scores_inner)
end   

mean = mean(scores_final)
mean_true = mean(tru_scores_final)
std_dev = std(scores_final)
std_dev_true = std(scores_final)

for i=1:80
var_count=0;
    for j=1:reps*cv_folds
        if coefs(j,i) ~= 0
            var_count = var_count + 1;
        end;
    end;
    if var_count>250
        i
    end;
end;
