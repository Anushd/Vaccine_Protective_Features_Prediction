function [result] = multisvm2(TrainingSet,GroupTrain,TestSet,tru_indices)

u=unique(GroupTrain);
numClasses=length(u);
result = zeros(length(TestSet(:,1)),1);
models = cell(1,(numClasses-1));

for k=fliplr(2:numClasses)
    G1vAll=(GroupTrain==u(k));
    Train_Sel = TrainingSet(:,tru_indices{k});
    models{k} = svmtrain(Train_Sel,G1vAll, 'kernel_function', 'rbf', 'rbf_sigma', 1.3);
    rm_k = (GroupTrain~=k);
    GroupTrain= GroupTrain(rm_k);
    TrainingSet= TrainingSet(rm_k,:);
end

%classify test cases
for j=1:size(TestSet,1)
    eq_fin_class = false;
    
    for k=fliplr(2:numClasses)
        if(svmclassify(models{k},TestSet(j,tru_indices{k}))) 
            break;
        elseif(~svmclassify(models{k},TestSet(j,tru_indices{k})) && k==2)
            eq_fin_class = true;
            break;
        end
    end
    
    if(eq_fin_class)
        result(j) = 1;
    else
        result(j) = k;
    end
end
