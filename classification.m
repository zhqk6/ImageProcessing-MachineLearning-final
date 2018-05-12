clear;
test_dataset=load('C:\Users\zhqk6\Desktop\new datas\test image representation1\All_test_representations.mat');
training_dataset.X=load('C:\Users\zhqk6\Desktop\new datas\image representation1\All_representations.mat');
%file path can be changed
label1=ones(30,1);
label2=2*ones(30,1);
%label1 is 1, label 2 is 2
training_dataset.label1=[label2;label1;label1;label1];
%airplane
training_dataset.label2=[label1;label2;label1;label1];
%face
training_dataset.label3=[label1;label1;label2;label1];
%leaves
training_dataset.label4=[label1;label1;label1;label2];
%motocycle

for i=1:120
    training_dataset.X.All_representations(i,:)=training_dataset.X.All_representations(i,:)/sum(training_dataset.X.All_representations(i,:));
end
for j=1:20
    test_dataset.All_test_representations(j,:)=test_dataset.All_test_representations(j,:)/sum(test_dataset.All_test_representations(j,:));
end
%Normalize

svmModel1=svmtrain(training_dataset.X.All_representations,training_dataset.label1,'kernel_function','rbf','rbf_sigma',2.5);
svmGroup1=svmclassify(svmModel1,test_dataset.All_test_representations);
% 100% accuracy result:1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 1
svmModel2=svmtrain(training_dataset.X.All_representations,training_dataset.label2,'kernel_function','rbf','rbf_sigma',3.5);
svmGroup2=svmclassify(svmModel2,test_dataset.All_test_representations);
% 100% accuracy result:1 1 1 1 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 1
svmModel3=svmtrain(training_dataset.X.All_representations,training_dataset.label3,'kernel_function','rbf','rbf_sigma',3);
svmGroup3=svmclassify(svmModel3,test_dataset.All_test_representations);
% 100% accuracy result:1 1 1 1 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1
svmModel4=svmtrain(training_dataset.X.All_representations,training_dataset.label4,'kernel_function','rbf','rbf_sigma',7);
svmGroup4=svmclassify(svmModel4,test_dataset.All_test_representations);
% 100% accuracy result:2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
%Classification by SVM


knnModel1=fitcknn(training_dataset.X.All_representations,training_dataset.label1,'NumNeighbors',5);
knngroup1=predict(knnModel1,test_dataset.All_test_representations);

knnModel2=fitcknn(training_dataset.X.All_representations,training_dataset.label2,'NumNeighbors',5);
knngroup2=predict(knnModel2,test_dataset.All_test_representations);

knnModel3=fitcknn(training_dataset.X.All_representations,training_dataset.label3,'NumNeighbors',4);
knngroup3=predict(knnModel3,test_dataset.All_test_representations);

knnModel4=fitcknn(training_dataset.X.All_representations,training_dataset.label4,'NumNeighbors',4);
knngroup4=predict(knnModel4,test_dataset.All_test_representations);
%Classification by KNN
