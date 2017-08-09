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
n_classes = 5;

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
        
        length_lab = length(ts_lab);
        tot_score = 0;
        
        %%%%
                    
        groupA_indices = find(tr_lab==1 | tr_lab==2);
        groupB_indices = find(tr_lab==3 | tr_lab==4);
        groupC_indices = find(tr_lab==5); 
        
        sel_tr_lab = zeros(size(tr_lab));           
        sel_tr_lab(groupA_indices) = 0;  
        sel_tr_lab(groupB_indices) = 1;
        sel_tr_lab(groupC_indices) = 2;
        
        sel_tr_lab_a = tr_lab(groupA_indices);
        sel_tr_dat_a = tr_dat(groupA_indices,:);
        sel_tr_lab_b = tr_lab(groupB_indices);
        sel_tr_dat_b = tr_dat(groupB_indices,:);
        
        sel_tr_dat = tr_dat;
            
        mdl = TreeBagger(100,sel_tr_dat,sel_tr_lab,'OOBPredictorImportance','on','MinLeafSize',5);
        mdl_a = TreeBagger(100,sel_tr_dat_a,sel_tr_lab_a,'OOBPredictorImportance','on','MinLeafSize',5);
        mdl_b = TreeBagger(100,sel_tr_dat_b,sel_tr_lab_b,'OOBPredictorImportance','on','MinLeafSize',5);
    
        pred = predict(mdl, ts_dat);
        pred = str2double(pred);
        
        %Check if predicted labels of 5 are correct
        pred_k_indices = find(pred==2);
        ts_lab_k_indices = find(ts_lab==5);
        score = numel(intersect(pred_k_indices,ts_lab_k_indices));
        tot_score = tot_score + score;

        for k=1:length_lab
            
            if pred(k)==0
                pred_a = predict(mdl_a, ts_dat(k,:));
                pred_a = str2double(pred_a);
                
                if ts_lab(k)==pred_a                   
                    tot_score = tot_score + 1;
                end;    
                
            elseif pred(k)==1
                pred_b = predict(mdl_b, ts_dat(k,:));
                pred_b = str2double(pred_b);
                
                if ts_lab(k)==pred_b
                    tot_score = tot_score + 1;   
                end;
            
            end;            
        end;
        
        tot_score = tot_score/length_lab;
        tru_scores_inner(1,i) = tot_score;        
        
    count = count+1;
    end
    tru_scores_final(1,j) = mean(tru_scores_inner)
end   

mean(tru_scores_final)
std(tru_scores_final)
