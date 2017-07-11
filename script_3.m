#stable feature selection

data = table2array(readtable('/Users/anush/documents/projects/ragon/data/Post_pass1_parsed.txt'));
train_data = data(:,2:81);
lab = data(:,83);
train_data = zscore(train_data);

coefs = zeros(50,80);
count = 1;
for j=1:100
    indices = crossvalind('Kfold', 46, 5);
    for i=1:5
        test = (indices == i); 
        train = ~test;
        ts_dat = train_data(test,:);
        ts_lab = lab(test,:);
        tr_dat = train_data(train,:);
        tr_lab = lab(train,:);
        [B, stats] = lasso(tr_dat,tr_lab, 'CV', 5);
        betas = B(:,stats.IndexMinMSE);
        coefs(count,:) =  betas';
        count = count+1;
    end
end   

for i=1:80
var_count=0;
    for j=1:500
        if coefs(j,i) ~= 0
            var_count = var_count + 1;
        end;
    end;
    if var_count>250
        i
    end;
end;
