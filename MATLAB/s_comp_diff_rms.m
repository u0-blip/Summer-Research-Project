%% importing the files
plot_shape = 'sphere';

num_para = 5;
dist = (1:num_para-1)/10;
dist = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
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
cube_size = 0.5;
cell_size = [2, 2, 2];
cell_lim = [-1 1;-1 1; -1 1];

cube_points = [[-1, -1, -1]; [-1, -1, 1]; [-1, 1, -1]; [-1, 1, 1]; [1, -1, -1]; [1, -1, 1]; [1, 1, -1]; [1, 1, 1]];
%normalize the cube points to 1
cube_points = cube_size/cell_size(1) * cube_points;


s = num2cell([2, size(cube_points)]);
cubes = zeros(s{:});

roi = [-0.15 0.15; -0.8 0.8;-1 1];

pos2index = @(pos, pos_range, index_range) (pos-pos_range(1))/(pos_range(2) - pos_range(1))*(index_range(2)-index_range(1))+index_range(1);

index_roi = roi;

for i = 1:3
    for j = 1:2
        index_roi(i,j) = pos2index(roi(i,j), cell_lim(i,:), [1, space_dim(2)]);
    end
end


set(gcf,'color','w');

for j = 1:8
    subplot(4, 2, j)
    
    plot_whole_field_rms = squeeze(arr_whole_field_rms(...
        j, index_roi(1,1):index_roi(1,2), ...
        index_roi(2,1):index_roi(2,2), ...
        index_roi(3,1):index_roi(3,2)));
    
    roi_dim = size(plot_whole_field_rms);
    
    [X,Y] = ndgrid(linspace(roi(2,1), roi(2,2), roi_dim(2)), ...
        linspace(roi(1,1), roi(1,2), roi_dim(1)));
    
    pc = pcolor(X', Y', squeeze(plot_whole_field_rms(:, :, roi_dim(2)/2)));
    
    hold on
    set(pc, 'EdgeColor', 'none'); 
    ax = gca;       
    
    if strcmp(plot_shape, 'sphere')
        radius = 0.1;
        center = [-dist(j)/2, 0; dist(j)/2, 0];
        plot_sphere(ax, center, radius)
    elseif strcmp(plot_shape, 'cube')
        center = [-dist(j)/2 - cube_size/2, 0, 0; dist(j)/2 + cube_size/2, 0, 0];
        plot_cube(ax, cube_points, center)
    end
    
    title('part dist = ' + string(dist(j)))
%     ax.TitleFontSizeMultiplier = 3;
    colorbar
%     caxis([0 6.5e-6])
end

    
