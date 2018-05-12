direc='C:\Users\zhqk6\Desktop\patches_train\descriptor(';
%direc can be changed 
Vectors128=zeros(128,172684);

sum1=0;
for i=1:120
    fn=strcat(direc,int2str(i-1));
    fn1=strcat(fn,')');
    img=imread(strcat(fn1,'.jpg'));
    [a(i),b]=size(img);
    for j=1:a(i)
        Vectors128(:,sum1+j)=img(j,:);
    end
    sum1=a(i)+sum1;
end
Vectors128=Vectors128';
[idx,c]=kmeans(Vectors128,20);
% centroids c is a dictionary with 20 words inside
% idx is the index of each descriptor(patch)

direc2='C:\Users\zhqk6\Desktop\new datas\image representation1\image(';
[a1,b1]=size(c);
All_representations=zeros(120,20);
All_test_representations=zeros(20,20);
 
 for i=1:120
    fn=strcat(direc,int2str(i-1));
    fn1=strcat(fn,')');
    img_patches=imread(strcat(fn1,'.jpg'));
    % for all the descriptors in each training image 
    hist=image_representation(img_patches,c);
    %generate the hist
    All_representations(i,:)=hist;
    figure(i);
    bar(1:a1,hist);
    fn4=strcat(direc2,int2str(i));fn5=strcat(fn4,').jpg');saveas(gcf,fn5);
    % show and save the representation histogram  
 end
 
 direc5='C:\Users\zhqk6\Desktop\test descriptor\descriptor(';
 direc6='C:\Users\zhqk6\Desktop\new datas\test image representation1\image(';
 %direc can be changed 
 
 for i=1:20
    fn6=strcat(direc5,int2str(i-1));
    fn7=strcat(fn6,')');
    test_img_patches=imread(strcat(fn7,'.jpg'));
     % for all the descriptors in each testing image 
    hist=image_representation(test_img_patches,c);
     %generate the hist
    All_test_representations(i,:)=hist;
    figure(i+120);
    bar(1:a1,hist);
    fn8=strcat(direc6,int2str(i));fn9=strcat(fn8,').jpg');saveas(gcf,fn9);
    % show and save the representation histogram  
 end