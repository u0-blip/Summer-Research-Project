dir = 'C:\peter_abaqus\Summer-Research-Project\meep\meep_out\';
name = 'test.bin';
fid = fopen(strcat(dir, name),'r');
test1 = fread(fid,'double','ieee-le');
fclose(fid);


%%
test1 = reshape(test1, 3,3);