%% importing the files
dir = 'C:\peter_abaqus\Summer-Research-Project\meep\meep_out\';
dist = 0.2;
plot_cube = 0;

name = strcat('cube_dis_', sprintf('%.1f',dist), '.bin');
name = 'voronoi_120_t_20_res_50_f_1.5_rms_5.bin';

[whole_field, space_dim] = impFile(name);

whole_field_rms = squeeze(rms(whole_field));

% imshow(rescale(squeeze(whole_field(1,1,:,:))))

%%
cube_size = 0.5;
cell_size = [2, 2, 2];

cube_points = [[-1, -1, -1]; [-1, -1, 1]; [-1, 1, -1]; [-1, 1, 1]; [1, -1, -1]; [1, -1, 1]; [1, 1, -1]; [1, 1, 1]];
%normalize the cube points to 1
cube_points = cube_size/cell_size(1) * cube_points;

s = num2cell([2, size(cube_points)]);
cubes = zeros(s{:});
cubes(1, :, :) = move_poly(cube_points, [-dist/2 - cube_size/2, 0, 0]);
cubes(2, :, :) = move_poly(cube_points, [dist/2 + cube_size/2, 0, 0]);

cubes_xy = cubes(:,:,[1,2]);
cubes_yz = cubes(:, :, [2, 3]);

range = space_dim(2);

figure(1)
set(gcf,'color','w');

roi = [0.1, 0.9];
len_roi = roi(2) - roi(1);
trans_roi = @(point, len_roi) ((point./cell_size + 1/2)/len_roi - (1/len_roi - 1)/2)*range*len_roi + 2;

if plot_cube
for i  = 1:size(cubes, 1)
    
    subplot(2, 2, 1)
    cube = squeeze(cubes(i, :, :));
    [k1,~] = convhull(cube);
    for j = 1:length(k1)
        p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
            plot3(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), trans_roi(p(:, 3), len_roi), 'r-')
        hold on
    end
end 
%     subplot(2, 2, 2)
%     
%     [k1,av1] = convhull(cube);
%     for j = 1:length(k1)
%         p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
%         plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
%         hold on
%     end
end

% xlim([-1, 1]) 
% ylim([-1, 1])
% zlim([-1, 1])

plot_limit = roi(1)*range:roi(2)*range;

if length(size(whole_field)) == 4
    plot_whole_field = whole_field;
    plot_whole_field = plot_whole_field(:, plot_limit, plot_limit, plot_limit);
    Q = size(plot_whole_field, 1);
    time_slice = squeeze(plot_whole_field(1,:,:,:));
elseif length(size(whole_field)) == 5
    plot_whole_field = squeeze(whole_field(1, :, :,:,:));
    plot_whole_field = plot_whole_field(:, plot_limit, plot_limit, plot_limit);
    Q = size(plot_whole_field, 1);
    time_slice = squeeze(plot_whole_field(1,:,:,:));
end
plot_whole_field_rms = whole_field_rms(plot_limit, plot_limit, plot_limit);

subplot(2, 2, 1)
time_slice = rid_val(time_slice);
h = slice(time_slice, [], 1:size(time_slice,3), []);
xlabel('x')
ylabel('y')
zlabel('z')

alpha(.1)
set(h, 'EdgeColor', 'none');
%set(h , 'FaceColor','interp');
colorbar
drawnow();
pause(0.3);
subplot(2,2,1)
title('3D spatial distribution of EM field')
subplot(2,2,2)
title('EM field distribution cut on xy plane')
subplot(2,2,3)
title('EM field distribution cut of xz plane')
subplot(2,2,4)
title('RMS EM field strength')

for K = 2 : Q
    time_slice = squeeze(plot_whole_field(K, :, :, :));
    
    subplot(2, 2, 2)
    pc = pcolor(squeeze(time_slice(1:34, :, range/2)));
    set(pc, 'EdgeColor', 'none');
    if plot_cube
    for i  = 1:size(cubes, 1)
        cube = squeeze(cubes(i, :, :));
        [k1,av1] = convhull(cube);
        for j = 1:length(k1)
            p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
            plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
            hold on
        end
    end
    end
    title('EM field distribution cut on xy plane')
    colorbar
    
    subplot(2, 2, 3)
    pc = pcolor(squeeze(time_slice(range/2,:,  :)));
    set(pc, 'EdgeColor', 'none');
    for i  = 1:size(cubes, 1)
        cube = squeeze(cubes(i, :, :));
        [k1,av1] = convhull(cube);
        if plot_cube
        for j = 1:length(k1)
            p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
            plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
            hold on
        end
        end
    end
    title('EM field distribution cut of xz plane')
    colorbar
    
    subplot(2, 2, 4)
    pc = pcolor(squeeze(plot_whole_field_rms(1:34, :, range/2)));
    set(pc, 'EdgeColor', 'none');
    for i  = 1:size(cubes, 1)
        cube = squeeze(cubes(i, :, :));
        [k1,av1] = convhull(cube);
        if plot_cube
        for j = 1:length(k1)
            p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
            plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
            hold on
        end
        end
    end
    title('RMS EM field strength')
    colorbar
    

    time_slice = rid_val(time_slice);
    sgtitle(strcat('time step :' , num2str(K)))
    for J = 1:length(h)
        set(h(J), 'CData', squeeze(time_slice( J, :, :)));
    end
    drawnow();
    pause(0.6);
end

function out_arr = rid_val(in_arr)
    out_arr = in_arr;
    arr_abs = abs(in_arr);
    arr_mean = mean(mean(mean(arr_abs)));
    out_arr(arr_abs <= (arr_mean)/10.)=nan;
end

function moved = move_poly(poly, move_vec)
    for i  = 1:length(poly)
        poly(i, :)= poly(i,:) + move_vec;
    end
    moved = poly;        
end
