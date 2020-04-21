function handle = plot_cube(handle, cube_points, center)
%PLOT_CUBE Summary of this function goes here
%   Detailed explanation goes here
    ran = [2,6,8,4,2];
    for i  = 1:size(center, 1)
        cube = move_poly(cube_points, center(i,:));
        plot(handle, cube(ran, 1), cube(ran, 2), 'r-');  
    end
end

