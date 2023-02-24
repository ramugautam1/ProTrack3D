function correlation20210408(Fullsize_1,Fullsize_2,Fullsize_regression_1,Fullsize_regression_2,t2,time,spatial_extend_matrix,addr2, padding)
depth=size(Fullsize_regression_1,4);
if ~exist('time','var')
     % third parameter does not exist, so default it to something
      time=1;
end

% get the size of sample
[x,y,z]=size(Fullsize_1);
[x_reserve,y_reserve,z_reserve]=size(Fullsize_1);
disp([x,y,z])
% padding the sample for 'extended search' (Fullsize: object label map, Fullsize_regression: object deep feature map)
Fullsize_1_padding=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2);
Fullsize_2_padding=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2);
Fullsize_regression_1_padding=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2,depth);
Fullsize_regression_2_padding=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2,depth);
Fullsize_1_padding(padding(1)+1:x+padding(1),padding(2)+1:y+padding(2),padding(3)+1:z+padding(3))=Fullsize_1;
Fullsize_2_padding(padding(1)+1:x+padding(1),padding(2)+1:y+padding(2),padding(3)+1:z+padding(3))=Fullsize_2;
Fullsize_regression_1_padding(padding(1)+1:x+padding(1),padding(2)+1:y+padding(2),padding(3)+1:z+padding(3),:)=Fullsize_regression_1(:,:,:,1:depth);
Fullsize_regression_2_padding(padding(1)+1:x+padding(1),padding(2)+1:y+padding(2),padding(3)+1:z+padding(3),:)=Fullsize_regression_2(:,:,:,1:depth);

%correlation_map_padding=zeros(x+padding*2, y+padding*2, z+4,max(max(max(Fullsize_1))));
correlation_map_padding_corr=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2);
correlation_map_padding_show=zeros(x+padding(1)*2, y+padding(2)*2, z+padding(3)*2);
clear Fullsize_regression_1 Fullsize_regression_2 Fullsize_1 Fullsize_2

Fullsize_1_label=Fullsize_1_padding;
% Fullsize_2_label=Fullsize_2_padding;
disp(max(max(max(Fullsize_1_padding))))
stats1 = regionprops3(Fullsize_1_padding,'BoundingBox','VoxelList','ConvexHull');
disp(size(stats1))
% stats2 = regionprops3(logical(Fullsize_2_padding),'BoundingBox','VoxelList','ConvexHull');
%filename = strcat(addr2,t2,'_tracking.xls');
disp('next')

% for each object
for i=1:height(stats1)
    if rem(i,50)==0
        disp(i/height(stats1))
    end
    % if the object is big, not processing all pixels of the object to save
    % time
    if size(stats1.VoxelList{i,1},1)<30
        stepsize=1;
    else
        stepsize=3;
    end
    % for a block of pixels in an object, seach for the most correlated nearby block
    % in previous time point
    for n1=1:stepsize:size(stats1.VoxelList{i,1},1)
        if stepsize==1
            index=n1;
        else
            index=ceil(rand()*size(stats1.VoxelList{i,1},1));
        end
%         if stats1.VoxelList{i,1}(index,2)<=x_reserve+padding && stats1.VoxelList{i,1}(index,1)<=y_reserve+padding && stats1.VoxelList{i,1}(index,3)<=z_reserve+2
            Feature_map1=Fullsize_regression_1_padding(stats1.VoxelList{i,1}(index,2)-3:stats1.VoxelList{i,1}(index,2)+3,stats1.VoxelList{i,1}(index,1)-3:stats1.VoxelList{i,1}(index,1)+3,stats1.VoxelList{i,1}(index,3)-1:stats1.VoxelList{i,1}(index,3)+1,:);
            % ---uncomment if the extended search decay is wanted
%             Feature_map1=Feature_map1.*spatial_extend_matrix;

            for m1=-1:1
                x=2*m1;
                for m2=-1:1
                    y=2*m2;
                    for m3=-1:1
                        z=m3;
                        Feature_map2=Fullsize_regression_2_padding(stats1.VoxelList{i,1}(index,2)+x-3:stats1.VoxelList{i,1}(index,2)+x+3,stats1.VoxelList{i,1}(index,1)+y-3:stats1.VoxelList{i,1}(index,1)+y+3,stats1.VoxelList{i,1}(index,3)+z-1:stats1.VoxelList{i,1}(index,3)+z+1,:);
                        
                                              
                        % ---uncomment if the extended search decay is wanted
                        %Feature_map2=Feature_map2.*spatial_extend_matrix;
                        %Feature_map1=Feature_map1/mean2(Feature_map1);
                        %Feature_map2=Feature_map2/mean2(Feature_map2);
                        %corr=convn(Feature_map1,Feature_map2(end:-1:1,end:-1:1,end:-1:1));
                        
                         %flattening the feature map
                        Feature_map1_flatten = Feature_map1(:);
                        Feature_map2_flatten = Feature_map2(:);
                        corr = corr2(Feature_map1_flatten,Feature_map2_flatten);
                        
                        if corr>0.2
                            b=stats1.VoxelList{i,1};
                            a=zeros;
                            for i1=1:size(b,1)
                                a(i1,1)=Fullsize_1_label(b(i1,2),b(i1,1),b(i1,3));
                            end
                            value=mode(a,'all');
                            countzero=size(a(a==0),1);
                            if countzero>value
                                value=0;
                            end
                            correlation_map_padding_corr_local=correlation_map_padding_corr(stats1.VoxelList{i,1}(index,2)+x-3:stats1.VoxelList{i,1}(index,2)+x+3,stats1.VoxelList{i,1}(index,1)+y-3:stats1.VoxelList{i,1}(index,1)+y+3,stats1.VoxelList{i,1}(index,3)+z-1:stats1.VoxelList{i,1}(index,3)+z+1);
                            correlation_map_padding_show_local=correlation_map_padding_show(stats1.VoxelList{i,1}(index,2)+x-3:stats1.VoxelList{i,1}(index,2)+x+3,stats1.VoxelList{i,1}(index,1)+y-3:stats1.VoxelList{i,1}(index,1)+y+3,stats1.VoxelList{i,1}(index,3)+z-1:stats1.VoxelList{i,1}(index,3)+z+1);
                            % only select the highest correlation and
                            % assign the label
                            correlation_map_padding_show_local(correlation_map_padding_show_local<corr)=value;
                            correlation_map_padding_corr_local(correlation_map_padding_corr_local<corr)=corr;
                            
                            correlation_map_padding_corr(stats1.VoxelList{i,1}(index,2)+x-3:stats1.VoxelList{i,1}(index,2)+x+3,stats1.VoxelList{i,1}(index,1)+y-3:stats1.VoxelList{i,1}(index,1)+y+3,stats1.VoxelList{i,1}(index,3)+z-1:stats1.VoxelList{i,1}(index,3)+z+1)=correlation_map_padding_corr_local;
                            correlation_map_padding_show(stats1.VoxelList{i,1}(index,2)+x-3:stats1.VoxelList{i,1}(index,2)+x+3,stats1.VoxelList{i,1}(index,1)+y-3:stats1.VoxelList{i,1}(index,1)+y+3,stats1.VoxelList{i,1}(index,3)+z-1:stats1.VoxelList{i,1}(index,3)+z+1)=correlation_map_padding_show_local;                                                       
                        end
                    end
                end
            end
%         end
    end
end
%niftiwrite(correlation_map_padding,strcat(addr2,'correlation_map_padding','_',t2,'.nii'));
disp(max(max(max(correlation_map_padding_show))))
disp(strcat(addr2,'correlation_map_padding_show_traceback',num2str(time),'_',t2,'.nii'))
niftiwrite(correlation_map_padding_show,strcat(addr2,'correlation_map_padding_show_traceback',num2str(time),'_',t2,'.nii'));
niftiwrite(correlation_map_padding_corr,strcat(addr2,'correlation_map_padding_hide_traceback',num2str(time),'_',t2,'.nii'));
end

