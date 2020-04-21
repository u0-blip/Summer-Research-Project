%% importing the files
plot_shape = 'cube';

num_para = 5;
dist = (1:num_para-1)/10;
dist = [0.2 0.3 0.4 0.5 0.6 0.7];
clear arr_whole_field_rms

for i = 1:length(dist)
    dir = 'C:\peter_abaqus\Summer-Research-Project\meep\meep_out\';
    
    name = strcat('cube_dis_', sprintf('%.1f',dist(i)), '.bin');

    whole_field = impFile(name);
    
    arr_whole_field_rms(i, :, :, :) = squeeze(rms(whole_field(6:end, :, :, :)));
end

space_dim = size(arr_whole_field_rms);
% imshow(rescale(squeeze(whole_field(1,1,:,:))))


%% 
figure()
cube_size = 0.4;
cell_size = [2, 2, 2];
cell_lim = [-1 1;-1 1; -1 1];

cube_points = [[-1, -1, -1]; [-1, -1, 1]; [-1, 1, -1]; [-1, 1, 1]; [1, -1, -1]; [1, -1, 1]; [1, 1, -1]; [1, 1, 1]];
%normalize the cube points to 1
cube_points = cube_size/cell_size(1) * cube_points;


s = num2cell([2, size(cube_points)]);
cubes = zeros(s{:});

roi = [-0.55 0.55; -0.8 0.8;-1 1];

pos2index = @(pos, pos_range, index_range) round((pos-pos_range(1))/(pos_range(2) - pos_range(1))*(index_range(2)-index_range(1))+index_range(1));

index_roi = roi;

for i = 1:3
    for j = 1:2
        index_roi(i,j) = pos2index(roi(i,j), cell_lim(i,:), [1, space_dim(2)]);
    end
end


set(gcf,'color','w');

num_g = 4;
for j = 1:num_g
    subplot(ceil(num_g/2), 2, j)
    
    plot_whole_field_rms = squeeze(arr_whole_field_rms(...
        j, index_roi(1,1):index_roi(1,2), ...
        index_roi(2,1):index_roi(2,2), ...
        index_roi(3,1):index_roi(3,2)));
    
    roi_dim = size(plot_whole_field_rms);
    
    [X,Y] = ndgrid(linspace(roi(2,1), roi(2,2), roi_dim(2)), ...
        linspace(roi(1,1), roi(1,2), roi_dim(1)));
    
    section = squeeze(plot_whole_field_rms(:, :, roi_dim(2)/2));

    pc = pcolor(X', Y', section);
    set(pc, 'EdgeColor', 'none'); 
    
    hold on
    ax = gca;       
    
    if strcmp(plot_shape, 'sphere')
        radius = 0.1;
        center = [-dist(j)/2, 0; dist(j)/2, 0];
        plot_sphere(ax, center, radius)
    elseif strcmp(plot_shape, 'cube')
        center = [ 0,-dist(j)/2 - cube_size/2+0.03, 0; 0, dist(j)/2 + cube_size/2+0.03, 0];
        plot_cube(ax, cube_points, center)
    end
    
    title('part dist = ' + string(dist(j)));
%     ax.TitleFontSizeMultiplier = 3;
    colorbar;
    caxis([1e-5 4e-5]);
end


  
%% plotting lateral and longtitudle line

% set(gcf,'color','w');
% graph_horizontal = 4;
% 
% num_g = 16;
% slices = linspace(20,40,num_g);
% slices = round(slices);
% count = 1;
% for j = slices
%     subplot(ceil(num_g/graph_horizontal), graph_horizontal, count)
%     count  = count+1;
%     section = squeeze(plot_whole_field_rms(:, :, j));
%     pc = pcolor(X', Y', section);
%     set(pc, 'EdgeColor', 'none'); 
%     title(string(j))
%     colorbar;
%     caxis([1e-5 4e-5]);
% end
% 

clear line_v
clear line_h
num_g = 6;
for j = 1:num_g
    line_v(j,:) = squeeze(arr_whole_field_rms(j, roi_dim(1)/2, :,roi_dim(2)/2));
    line_h(j,:) = squeeze(arr_whole_field_rms(j, :, roi_dim(2)/2,roi_dim(2)/2));
    leng_v(j,:) = 'cube dist: ' + string(j/10) + ' vertical';
    leng_h(j,:) = 'cube dist: ' + string(j/10) + ' horizontal';
    
end

figure()
set(gcf,'color','w');
subplot(1,2,1)
plot(line_v')
legend(leng_v)
title('verticle cut EM strength')
axis([-inf, inf, 0.5e-5, 1.2e-5])

subplot(1,2,2)
plot(line_h')
legend(leng_h, 'location', 'northwest')
title('horizontal cut EM strength')
axis([-inf, inf, 0, 5e-5])