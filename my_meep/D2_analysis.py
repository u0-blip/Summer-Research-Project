
## importing the files
plot_shape='cube'
num_para=5
clear('eps')
# dist = [0.2 0.3 0.4 0.5 0.6 0.7];
# 001 050 100 150 200 250 
# dist = [0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6]
# dist = [30 40 50 60 70 80];
# dist = [0.2 0.3 0.4 0.5 0.6 0.7]-0.1;
# dist = [4 5 6 7 8 9 10];
# dist_vertical_cube = [0.25 0.27 0.30 0.35 0.37];
    wavelength=concat([1,40,100,150,200,250])
    dist=copy(dist)
    clear('arr_whole_field_rms')
    title_='source_size_0_8'
    mag_range=20
    lower=- 43
    shift_up=10
    for i in arange(1,2).reshape(-1):
        dir='C:\\peter_abaqus\\Summer-Research-Project\\meep\\meep_out\\'
        #     name = strcat('cube2D_dist_', sprintf('#.1f',dist(i)), '.bin');
#     name = strcat('sphere2D_dist_s0.4_', sprintf('#.1f',dist(i)), '.bin');
        name=strcat(title_,'.bin')
        #     name = strcat('sphere2D_div_', sprintf('#d',dist(i)), '.bin');
#     name = strcat('cube2D_width_', sprintf('#.1f',dist(i)), '.bin');
#     name = strcat('sphere2D_div1D_', sprintf('#d',dist(i)), '.bin');
#     name = strcat('cube2D_vertical_', sprintf('#.2f',dist_vertical_cube(i)), '.bin');
#     name = strcat('cube_source_wavelen_', sprintf('#03d',wavelength(i)), '.bin');
        eps_name=strcat(name,'.eps')
        clear('whole_field')
        clear('single_field')
        clear('single_eps')
        single_field=impFile(dir,name)
        if size(size(single_field),2) == 3:
            whole_field[i,arange(),arange(),arange()]=single_field
            single_eps=edge(impFile(dir,eps_name))
            eps[i,arange(),arange()]=single_eps
        else:
            #         squeeze(mean(single_field(:, :,:, 46:86), 4))
#         round(size(single_field,4)/2)
            whole_field[i,arange(),arange(),arange()]=squeeze(single_field(arange(),arange(),arange(),round(size(single_field,4) / 2)))
            single_eps=impFile(dir,eps_name)
            single_eps=edge(squeeze(single_eps(arange(),arange(),round(size(single_eps,3) / 2) + 10)))
            eps[i,arange(),arange()]=single_eps
        arr_whole_field_rms[i,arange(),arange()]=squeeze(rms(whole_field(i,arange(end() - 350,end()),arange(),arange())))
    
    space_dim=size(arr_whole_field_rms)
    # imshow(rescale(squeeze(whole_field(1,1,:,:))))
    
    ##
    figure()
    cell_size=concat([2,2,2])
    cell_lim=concat([[- 1,1],[- 1,1],[- 1,1]])
    cube_points=concat([[concat([- 1,- 1,- 1])],[concat([- 1,- 1,1])],[concat([- 1,1,- 1])],[concat([- 1,1,1])],[concat([1,- 1,- 1])],[concat([1,- 1,1])],[concat([1,1,- 1])],[concat([1,1,1])]])
    #normalize the cube points to 1
    cube_points=dot(cube_size / cell_size(1),cube_points)
    s=num2cell(concat([2,size(cube_points)]))
    cubes=zeros(s[arange()])
    roi=concat([[- 1,1],[- 1,1],[- 1,1]])
    pos2index=lambda pos=None,pos_range=None,index_range=None: round(dot((pos - pos_range(1)) / (pos_range(2) - pos_range(1)),(index_range(2) - index_range(1))) + index_range(1))
    index_roi=copy(roi)
    for i in arange(1,3).reshape(-1):
        for j in arange(1,2).reshape(-1):
            index_roi[i,j]=pos2index(roi(i,j),cell_lim(i,arange()),concat([1,space_dim(2)]))
    
    set(gcf,'color','w')
    num_column=1
    num_g=1
    for j in arange(1,num_g).reshape(-1):
        subplot(ceil(num_g / 2),num_column,j)
        #     plot_whole_field_rms = squeeze(arr_whole_field_rms(...
#         j, index_roi(1,1):index_roi(1,2), ...
#         index_roi(2,1):index_roi(2,2), ...
#         index_roi(3,1):index_roi(3,2)));
        dim=size(arr_whole_field_rms)
        #     [X,Y] = ndgrid(linspace(-cell_size(1)/2, cell_size(1)/2, dim(2)), ...
#         linspace(-cell_size(2)/2, cell_size(2)/2, dim(3)));
        outline=squeeze(eps(j,arange(),arange()))
        data=squeeze(arr_whole_field_rms(j,arange(),arange()))
        #     strength_profile_x(j,:) = data(135, :);
#     strength_profile_y(j,:) = data(:,133);
#     leng_v(j,:) = 'cube width: ' + string(j/10) + ' vertical';
#     leng_h(j,:) = 'cube width: ' + string(j/10) + ' horizontal';
        data=mag2db(data)
        data[outline]=max(max(data))
        pc=pcolor(data)
        hold('on')
        set(pc,'EdgeColor','none')
        ax=copy(gca)
        #     if strcmp(plot_shape, 'sphere')
#         radius = 0.1;
#         center = [-dist(j)/2, 0; dist(j)/2, 0];
#         plot_sphere(ax, center, radius);
#     elseif strcmp(plot_shape, 'cube')
#         center = [ -dist(j)/2 - cube_size/2+0.03, 0,0; dist(j)/2 + cube_size/2+0.03,0,  0];
#         plot_cube(ax, cube_points, center);
#     end
        #     title('part width = ' + string(dist(j)));
        title(strrep(title_,'_',' '))
        #     ax.TitleFontSizeMultiplier = 3;
        colorbar
        #     caxis([3e-3 10e-3]);
        caxis(concat([lower,lower + mag_range]) + shift_up)
        #     axis([cell_size(1)/2.0*roi(2,:) , cell_size(2)/2.*roi(1,:)])
#     axis([75, 250, 130, 210])
    
    
    ## plotting lateral and longtitudle line
    plt.figure()
#     set(gcf,'color','w')
    plt.subplot(1,2,1)
    plt.plot(strength_profile_x.T)
    plt.legend(leng_h,'location','south')
    plt.title('horizontal cut EM strength')
    plt.subplot(1,2,2)
    plt.plot(strength_profile_y.T)
    plt.legend(leng_v,'location','northwest')
    plt.title('vertical cut EM strength')