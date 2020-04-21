def impFile(dir=None,name=None,*args,**kwargs):
    varargin = impFile.varargin
    nargin = impFile.nargin

    
    fid=fopen(strcat(dir,name,'.meta'),'r')
# ..\impFile.m:3
    space_dim=fread(fid,'double','ieee-le')
# ..\impFile.m:4
    fclose(fid)
    fid=fopen(strcat(dir,name),'r')
# ..\impFile.m:7
    whole_field=fread(fid,'double','ieee-le')
# ..\impFile.m:8
    fclose(fid)
    if length(space_dim) == 2:
        whole_field=reshape(whole_field,space_dim(1),length(whole_field) / space_dim(1))
# ..\impFile.m:13
    elif length(space_dim) == 3:
        whole_field=reshape(whole_field,space_dim(1),space_dim(2),length(whole_field) / (dot(space_dim(1),space_dim(2))))
# ..\impFile.m:15
    elif length(space_dim) == 4:
        whole_field=reshape(whole_field,space_dim(1),space_dim(2),space_dim(3),length(whole_field) / (dot(dot(space_dim(1),space_dim(2)),space_dim(3))))
# ..\impFile.m:17
    elif length(space_dim) == 5:
        whole_field=reshape(whole_field,space_dim(1),space_dim(2),space_dim(3),space_dim(4),length(whole_field) / (dot(dot(dot(space_dim(1),space_dim(2)),space_dim(3)),space_dim(4))))
# ..\impFile.m:19
    
    return whole_field,space_dim
    
if __name__ == '__main__':
    pass
    