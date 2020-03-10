function handle = plot_sphere(handle, center, radius)
    t = 0:0.1:3*pi;
    for k =1:size(center,1)
        x = radius*sin(t) + center(k, 1);
        y = radius*cos(t) + center(k, 2);
        plot(handle, x, y);
    end
end