%% importing the files
plot_shape = 'cube';

num_para = 5;

clear eps
dist = [0.2 0.3 0.4 0.5 0.6 0.7];
% 001 050 100 150 200 250 
% dist = [0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6]
% dist = [30 40 50 60 70 80];
% dist = [0.2 0.3 0.4 0.5 0.6 0.7]-0.1;
% dist = [4 5 6 7 8 9 10];
% dist_vertical_cube = [0.25 0.27 0.30 0.35 0.37];
wavelength = [001 050 100 150 200 250 ];

dist = dist;

clear arr_whole_field_rms

title_ = 'temp\temp';


for i = 1:1
    dir = 'C:\peter_abaqus\Summer-Research-Project\data\';
    
%     name = strcat('cube2D_dist_', sprintf('%.1f',dist(i)), '.bin');
%     name = strcat('sphere2D_dist_s0.4_', sprintf('%.1f',dist(i)), '.bin');
        name = strcat(title_,  '.mpout');
%     name = strcat('sphere2D_div_', sprintf('%d',dist(i)), '.bin');
%     name = strcat('cube2D_width_', sprintf('%.1f',dist(i)), '.bin');
%     name = strcat('sphere2D_div1D_', sprintf('%d',dist(i)), '.bin');
%     name = strcat('cube2D_vertical_', sprintf('%.2f',dist_vertical_cube(i)), '.bin');
%     name = strcat('cube_source_wavelen_', sprintf('%03d',wavelength(i)), '.bin');

    eps_name = strcat('temp\temp.mpout', '.eps');
    clear whole_field
    clear single_field
    clear single_eps
    
    single_field = impFile(dir, name);
    
    if size(size(single_field),2) == 3
        whole_field(i, :,:,:) = single_field;
        single_eps = edge(impFile(dir, eps_name));
        eps(i, :, :) = single_eps;
    else
%         squeeze(mean(single_field(:, :,:, 46:86), 4))
%         round(size(single_field,4)/2)
        whole_field(i, :,:,:) = squeeze(single_field(:, :,:, round(size(single_field,4)/2)));
        single_eps = impFile(dir, eps_name);
        single_eps = edge(squeeze(single_eps(:,:,round(size(single_eps,3)/2)+10)));
        eps(i, :, :) = single_eps;
    end
    
    
    arr_whole_field_rms(i, :, :) = squeeze(trapz((whole_field(i, 1:30, :, :).^2)));
    
end

space_dim = size(arr_whole_field_rms);
% imshow(rescale(squeeze(whole_field(1,1,:,:))))


%% 


down_reduced_by = 200;
up_reduced_by = 10;

far_slice = round(space_dim(2)*0.75);
near_slice = round(space_dim(2)*0.15);
    
figure()
cell_size = [2, 2, 2];
cell_lim = [-1 1;-1 1; -1 1];

roi = [-1 1; -1 1;-1 1];


pos2index = @(pos, pos_range, index_range) round((pos-pos_range(1))/(pos_range(2) - pos_range(1))*(index_range(2)-index_range(1))+index_range(1));

index_roi = roi;

for i = 1:3
    for j = 1:2
        index_roi(i,j) = pos2index(roi(i,j), cell_lim(i,:), [1, space_dim(2)]);
    end
end


set(gcf,'color','w');
num_column = 1;

num_g = 1;
for j = 1:num_g
    subplot(ceil(num_g/2), num_column, j)
    
%     plot_whole_field_rms = squeeze(arr_whole_field_rms(...
%         j, index_roi(1,1):index_roi(1,2), ...
%         index_roi(2,1):index_roi(2,2), ...
%         index_roi(3,1):index_roi(3,2)));
    
    dim = size(arr_whole_field_rms);
    
%     [X,Y] = ndgrid(linspace(-cell_size(1)/2, cell_size(1)/2, dim(2)), ...
%         linspace(-cell_size(2)/2, cell_size(2)/2, dim(3)));

    outline = squeeze(eps(j,:,:));
    data = squeeze(arr_whole_field_rms(j, :, :));
    
%     strength_profile_x(j,:) = data(135, :);
%     strength_profile_y(j,:) = data(:,133);
%     leng_v(j,:) = 'cube width: ' + string(j/10) + ' vertical';
%     leng_h(j,:) = 'cube width: ' + string(j/10) + ' horizontal';
    
%     data = mag2db(data);
    og_data = data;
    
    outline2d = (sum(outline, 1) > 0.1);
    outline2d(outline2d) = max(max(data));
    

    
    data(outline) = max(max(data))*1.3;
    data(far_slice, :) = max(max(data));
    data(near_slice, :) = min(min(data));
    pc = pcolor(data);
    

    
   
    hold on
    
    set(pc, 'EdgeColor', 'none'); 
    ax = gca;       
    
%     if strcmp(plot_shape, 'sphere')
%         radius = 0.1;
%         center = [-dist(j)/2, 0; dist(j)/2, 0];
%         plot_sphere(ax, center, radius);
%     elseif strcmp(plot_shape, 'cube')
%         center = [ -dist(j)/2 - cube_size/2+0.03, 0,0; dist(j)/2 + cube_size/2+0.03,0,  0];
%         plot_cube(ax, cube_points, center);
%     end
    
%     title('part width = ' + string(dist(j)));
    title(strrep(title_, '_', ' '));
%     ax.TitleFontSizeMultiplier = 3;
    colorbar;
    if min(min(og_data)) < -600
        low = -600;
    end
  
%     caxis([0, 7.3e-5]);
%        caxis([0, 3e-3]);
%     caxis([lower, lower+mag_range]+shift_up);
%     axis([cell_size(1)/2.0*roi(2,:) , cell_size(2)/2.*roi(1,:)])
%     axis([65, 85, 120, 130])
    
    clear far_field
    clear near_field
    
    far_slice = 200;
    far_field = og_data(:, far_slice );
    near_field = og_data(near_slice,: );
end

time_dom = 0;

if time_dom
    far_field = squeeze(single_field(:, 205, 210));
end

figure()
set(gcf,'color','w');

plot(far_field)
% axis([-inf, inf, 0, 6e-5])
title('far field strength')



figure()
set(gcf,'color','w');
freq = abs(fft(far_field));
Fs=40;
n = length(freq);
f = Fs*(0:(n/2))/n;
P = abs(freq/n);

plot(f,P(1:n/2+1)) 
title('Gaussian Pulse in Frequency Domain')
xlabel('Frequency (f)')
ylabel('|P(f)|')

% hold on
% plot(near_field)
% title('near field strength')
  
% %% plotting lateral and longtitudle line
% figure()
% set(gcf,'color','w');
% 
% 
% subplot(1,2,1)
% plot(strength_profile_x');
% legend(leng_h, 'location', 'south')
% title('horizontal cut EM strength')
% 
% 
% subplot(1,2,2)
% plot(strength_profile_y');
% legend(leng_v, 'location', 'northwest')
% title('vertical cut EM strength')





