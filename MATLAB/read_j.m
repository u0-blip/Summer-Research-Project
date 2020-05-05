figure()
fid = fopen('C:\peter_abaqus\Summer-Research-Project\test_fortran\v_500_s_151_single.mpout','r');

scale = 4;
data = fread(fid,'double','ieee-le');
data = reshape(data, 90*scale, 90*scale);

pc = pcolor(data);
set(gcf,'color','w');
set(pc, 'EdgeColor', 'none');
axis([92, 106, 108, 126])

colorbar;
fclose(fid);
