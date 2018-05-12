function hist=image_representation(input_patch,centroid)
% used for generate a vector of representation histogram
[a,~]=size(input_patch);
[a1,~]=size(centroid);
hist=zeros(1,a1);
for i=1:a
    [~,I(i)]=pdist2(centroid,input_patch(i,:),'euclidean','Smallest',1);
    %calculating the euclidean distance and returning the smallest one
end
for i=1:a1
    hist(i)=length(find(I==i));
    %find the length of each k.
end