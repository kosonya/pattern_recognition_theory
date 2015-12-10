curfile = mfilename('fullpath');
curfile_split = strsplit(curfile, filesep);
cur_folder = curfile_split(1:end-1);
root_folder = curfile_split(1:end-2);

fnames = {'test.filtered.nandrop', 'test.filtered.nandrop.normalized', 'test.filtered.normalized', 'train.filtered.nandrop', 'train.filtered.nandrop.normalized', 'train.filtered.normalized'};

for i = 1:size(fnames, 2)
    fname = fnames{i};
    disp(fname);
    disp('Loading');
    csv_fname = strjoin({fname, 'csv'}, '.');
    csv_fname = strjoin([root_folder, csv_fname], filesep);

    training_data = csvread(csv_fname);

    mat_fname = strjoin({fname, 'mat'}, '.');
    mat_fname = strjoin([root_folder, mat_fname], filesep);
    disp('Saving');
    save(mat_fname, 'training_data', '-v7.3');
end