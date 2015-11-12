curfile = mfilename('fullpath');
curfile_split = strsplit(curfile, filesep);
cur_folder = curfile_split(1:end-1);
root_folder = curfile_split(1:end-2);

csv_fname = strjoin([root_folder, 'train_filtered.csv'], filesep);

training_data = csvread(csv_fname);

mat_fname = strjoin([root_folder, 'training_data.mat'], filesep);

save(mat_fname, 'training_data');