
%rng(124)
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

features_select_lasso = zeros(reps*cv_folds,n_features);
features_select_rf = zeros(reps*cv_folds,n_features); 

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
           
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==2 | tr_lab==3 | tr_lab==4 | tr_lab==5);
        ind2 = find(tr_lab==1);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        betas1 = B(:,stats.IndexMinMSE-1)';
        
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==1 | tr_lab==3 | tr_lab==4 | tr_lab==5);
        ind2 = find(tr_lab==2);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        betas2 = B(:,stats.IndexMinMSE-1)';
        
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==1 | tr_lab==2 | tr_lab==4 | tr_lab==5);
        ind2 = find(tr_lab==3);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        betas3 = B(:,stats.IndexMinMSE-1)';
       
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==1 | tr_lab==2 | tr_lab==3 | tr_lab==5);
        ind2 = find(tr_lab==4);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        betas4 = B(:,stats.IndexMinMSE-1)';
        
        %%%%%%%%%%%%%%%%%%%%%%
        ind1 = find(tr_lab==1 | tr_lab==2 | tr_lab==3 | tr_lab==4);
        ind2 = find(tr_lab==5);
        tr_lab_lasso=tr_lab;
        tr_dat_lasso=tr_dat;      
        tr_lab_lasso(ind1)=1;
        tr_lab_lasso(ind2)=0;   
        tr_lab_lasso = tr_lab_lasso([ind1;ind2]);
        tr_dat_lasso = tr_dat_lasso([ind1;ind2],:);
        
        [B, stats] = lasso(tr_dat_lasso,tr_lab_lasso, 'CV', 5, 'Alpha', 0.5);
        betas5 = B(:,stats.IndexMinMSE-1)';
        
        %%%%%%%%%%%%%%%%%%%%%%
        
        betas = sort(unique([find(betas1~=0),find(betas2~=0),find(betas3~=0),find(betas4~=0),find(betas5~=0)]),'ascend');
        tr_dat = tr_dat(:,betas);
        ts_dat = ts_dat(:,betas);

        size(tr_dat);
        
        %%%%
        data = [tr_dat,tr_lab];
        [I_Cx, I_Cxx, I_xx, H_x, H_xx, H_C] = Fuzzy_MI(data);        
        [C,I] = sort(I_Cxx(:),'descend');
        [I_row,I_col] = ind2sub(size(I_Cxx),I(1:75)); %500,150
        
        %rm = find(H_x>prctile(H_x, 15)); %15
        %betas = find(I_Cx>0.35);
        %size(betas)
        
        tr_dat = tr_dat(:,unique([I_row, I_col]));
        ts_dat = ts_dat(:,unique([I_row, I_col]));
        
        size(tr_dat);
        %%%%
        
        %%%%
        data = [tr_dat,tr_lab];
        [I_Cx, I_Cxx, I_xx, H_x, H_xx, H_C] = Fuzzy_MI(data);       
        %rm = find(H_x>prctile(H_x, 30) & H_x<prctile(H_x, 70));
        rm = find(H_x<1e10);
        betas = find(I_Cx>0.35); %0.35
        betas = intersect(rm,betas);
        tr_dat = tr_dat(:,betas);
        ts_dat = ts_dat(:,betas);
        
        size(tr_dat);
        features_select_lasso(count,betas) = 1;
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
