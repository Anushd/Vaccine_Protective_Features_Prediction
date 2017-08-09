
%rng(100)
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

scores_final = zeros(1,reps);
tru_scores_final = zeros(1,reps);

coefs = zeros(reps*cv_folds,n_features);
count = 1;

features_select_lasso = zeros(reps*cv_folds,n_features);
features_select_rf = zeros(reps*cv_folds,n_features);

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
        
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==1 | tr_lab==2);
        ind2 = find(tr_lab==3 | tr_lab==4);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);        
        if stats.IndexMinMSE>1
            betas1 = B(:,stats.IndexMinMSE-1)';
        else
            fprintf('-1');
            betas1 = B(:,stats.IndexMinMSE)';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        ind1 = find(tr_lab==1);
        ind2 = find(tr_lab==2);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        if stats.IndexMinMSE>2
            betas2 = B(:,stats.IndexMinMSE-2)';
        elseif stats.IndexMinMSE>1
            fprintf('-1');
            betas2 = B(:,stats.IndexMinMSE-1)';
        else
            fprintf('-2');
            betas2 = B(:,stats.IndexMinMSE)';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        ind1 = find(tr_lab==3);
        ind2 = find(tr_lab==4);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        if stats.IndexMinMSE>2
            betas3 = B(:,stats.IndexMinMSE-2)';
        elseif stats.IndexMinMSE>1
            fprintf('-1');
            betas3 = B(:,stats.IndexMinMSE-1)';
        else
            fprintf('-2');
            betas3 = B(:,stats.IndexMinMSE)';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        ind1 = find(tr_lab==1 | tr_lab==2);
        ind2 = find(tr_lab==5);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.2);
        betas4 = B(:,stats.IndexMinMSE-1)';
        
        %%%%%%%%%%%%%%%%%%%%%%  
        
        ind1 = find(tr_lab==3 | tr_lab==4);
        ind2 = find(tr_lab==5);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.2);
        betas5 = B(:,stats.IndexMinMSE)';
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        betas = sort(unique([find(betas1~=0),find(betas2~=0),find(betas3~=0),find(betas4~=0),find(betas5~=0)]),'ascend');
        tr_dat = tr_dat(:,betas);
        ts_dat = ts_dat(:,betas);
        
        features_select_lasso(count,betas) = 1;
        
        %%%%
        
        %data = [tr_dat,tr_lab];
        %for k=1:10
        %    [data, ind] = feat_sel_sim(data,'luca',5);
        %    ts_dat(:,ind) = [];
        %end;
        %tr_dat = data;
        %tr_dat(:,end) = [];
        
        %%%%
        
        mdl = TreeBagger(200,tr_dat,tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        %mdl = fitcecoc(tr_dat, tr_lab);
        %mdl = multisvm(tr_dat, tr_lab, ts_dat);
        %mdl=fitctree(tr_dat, tr_lab);
        
        imp = mdl.OOBPermutedPredictorDeltaError;
        features_select_rf(count,betas) = imp;
        
        pred = predict(mdl, ts_dat);
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
