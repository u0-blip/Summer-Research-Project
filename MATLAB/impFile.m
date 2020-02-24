function [whole_field, space_dim] = impFile(name)
    dir = 'C:\peter_abaqus\Summer-Research-Project\meep\meep_out\';
    
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
end


