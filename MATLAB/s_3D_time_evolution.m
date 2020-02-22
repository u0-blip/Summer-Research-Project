%% importing the files
fid = fopen('C:\peter_abaqus\Summer-Research-Project\fortran_read\out_small.peter','r');
a = fread(fid,'double','ieee-le');
space_dim = [100, 100, 100];
a = reshape(a, space_dim(1), space_dim(2), space_dim(3), length(a)/(space_dim(1)*space_dim(2)*space_dim(3)));
fclose(fid);

%% importing the files
fid = fopen('C:\peter_abaqus\Summer-Research-Project\working_with_meep\ez_straight_waveg.bin','r');
a = fread(fid,'double','ieee-le');
space_dim = [201, 160, 80]; 
a = reshape(a, space_dim(1), space_dim(2), space_dim(3), length(a)/(space_dim(1)*space_dim(2)*space_dim(3)));
fclose(fid);
imshow(rescale(squeeze(a(1,:,:))))


%% importing the files
dir = 'C:\peter_abaqus\Summer-Research-Project\meep\meep_out\';
dist = 0.5;
name = strcat('cube_dis_', sprintf('%.1f',dist), '.bin');

fid = fopen(strcat(dir, name, '.meta'),'r');
space_dim = fread(fid,'double','ieee-le');
fclose(fid);

fid = fopen(strcat(dir, name),'r');
whole_field = fread(fid,'double','ieee-le');
fclose(fid);

% fid = fopen('C:\peter_abaqus\Summer-Research-Project\working_with_meep\one_cube_3d.bin','r');
% whole_field_2 = fread(fid,'double','ieee-le');
% fclose(fid);

% whole_field = whole_field -  whole_field_2;

if length(space_dim) == 3
    whole_field = reshape(whole_field, space_dim(1), space_dim(2), length(whole_field)/(space_dim(1)*space_dim(2)));
elseif length(space_dim) == 4 
    whole_field = reshape(whole_field, space_dim(1), space_dim(2), space_dim(3), length(whole_field)/(space_dim(1)*space_dim(2)*space_dim(3)));
elseif length(space_dim) == 5
    whole_field = reshape(whole_field, space_dim(1), space_dim(2), space_dim(3),space_dim(4), length(whole_field)/(space_dim(1)*space_dim(2)*space_dim(3)*space_dim(4)));
end

% imshow(rescale(squeeze(whole_field(1,1,:,:))))

%% export the file 

%% plotting options
plot_coutour=1;

%% plot iso-surface

b = a(:,:,:,1);
M = mean( b , 'all' );
isosurface(b, M)


%% plotting the time serie values
figure()
part_field = squeeze(whole_field(:, :, :, 50));

Q = size(part_field, 1);
W = squeeze(part_field(1,:,:));
h = pcolor(W);
h.FaceColor = 'interp';
set(h, 'EdgeColor', 'none');
drawnow();
pause(0.3);
for K = 2 : Q
    W = squeeze(part_field(K,:,:));
    set(h, 'CData', W);
    drawnow();
    pause(0.3);
end



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
    
roi = [0.1, 0.9];
len_roi = roi(2) - roi(1);
trans_roi = @(point, len_roi) ((point./cell_size + 1/2)/len_roi - (1/len_roi - 1)/2)*range*len_roi + 1;

for i  = 1:size(cubes, 1)
    
    subplot(1, 2, 1)
    cube = squeeze(cubes(i, :, :));
    [k1,~] = convhull(cube);
    for j = 1:length(k1)
        p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
        plot3(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), trans_roi(p(:, 3), len_roi), 'r-')
        hold on
    end
    
    subplot(1, 2, 2)
    
    [k1,av1] = convhull(cube);
    for j = 1:length(k1)
        p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
        plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
        hold on
    end
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

subplot(1, 2, 1)
h = slice(rid_val(time_slice), [], 1:size(time_slice,3), []);
xlabel('x')
ylabel('y')
zlabel('z')

alpha(.1)
set(h, 'EdgeColor', 'none');
%set(h , 'FaceColor','interp');
colorbar
drawnow();
pause(0.3);


for K = 2 : Q
    time_slice = squeeze(plot_whole_field(K, :, :, :));
    
    subplot(1, 2, 2)
    pc = pcolor(squeeze(time_slice(:, :, range/2)));
    set(pc, 'EdgeColor', 'none');
    
    for i  = 1:size(cubes, 1)
        
        cube = squeeze(cubes(i, :, :));
        [k1,av1] = convhull(cube);
        for j = 1:length(k1)
            p = cube([int32(k1(j,:)), int32(k1(j, 1))], :);
            plot(trans_roi(p(:, 1), len_roi), trans_roi(p(:, 2), len_roi), 'r-')
            hold on
        end
    end
    
    time_slice = rid_val(time_slice);
    title(K)
    for J = 1:length(h)
        set(h(J), 'CData', squeeze(time_slice( J, :, :)));
    end
    drawnow();
    pause(0.6);
    colorbar
end