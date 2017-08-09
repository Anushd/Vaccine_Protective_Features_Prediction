%Class A: elite controllers; Class B: cart treated & chronic untreated

data1 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_array_raw.txt'));
data2 = table2array(readtable('/Users/anush/documents/projects/ragon/HIV/data/Filtered_functionalglycan_raw.txt'));

lab = data1(:,1);
group1_indices = find(lab==1);
group34_indices = find(lab==3 | lab==4);
lab(group34_indices) = 0;
lab = lab(sort([group1_indices; group34_indices]));

train_data = [data1(:,2:313),data2(:,2:34)];
train_data = zscore(train_data);
train_data = train_data(sort([group1_indices; group34_indices]),:);

reps= 10;
cv_folds = 5;
n_samples=41;
n_features = 345;

indices = [16,179,200];

idx = kmeans(train_data(:,indices), 2);
results = [lab,idx]
