
% Reads the data from the initial data file and creates a data file
% containing only the two types of images we want.


fid = fopen('..\data2\smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r');
% fid = fopen('..\data2\smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r');
fread(fid,4,'uchar'); % result = [85 76 61 30], byte matrix(in base 16: [55 4C 3D 1E])
fread(fid,4,'uchar'); % result = [4 0 0 0], ndim = 4
fread(fid,4,'uchar'); % result = [236 94 0 0], dim0 = 24300 (=94*256+236)
fread(fid,4,'uchar'); % result = [2 0 0 0], dim1 = 2
fread(fid,4,'uchar'); % result = [96 0 0 0], dim2 = 96
fread(fid,4,'uchar'); % result = [96 0 0 0], dim3 = 96

fid2 = fopen('..\data2\subsetValTest.mat', 'w+');
for i = 1:4860
    
t0 = fread(fid,96*96);
t1 = fread(fid,96*96);
t2 = fread(fid,96*96);
t3 = fread(fid,96*96);
t4 = fread(fid,96*96);
t5 = fread(fid,96*96);
t6 = fread(fid,96*96);
t7 = fread(fid,96*96);
t8 = fread(fid,96*96);
t9 = fread(fid,96*96);

fwrite(fid2,t2);
fwrite(fid2,t3);
fwrite(fid2,t8);
fwrite(fid2,t9);
end
fclose(fid);
fclose(fid2);
