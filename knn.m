curfile = mfilename('fullpath');
curfile_split = strsplit(curfile, filesep);
root_folder = curfile_split(1:end-2);

mat_fname = strjoin([root_folder, 'test.filtered.nandrop.normalized.mat'], filesep);
fprintf('loading %s\n', mat_fname);
tic;
load(mat_fname);
toc;

disp('Loaded, splitting');
tic;
features = training_data(:,3:end-1);
labels = training_data(:,end);
toc;

disp('Running PCA');
tic;
[COEFF,SCORE] = princomp(features, 'econ');
features = SCORE(:,1:10);
toc;

disp('Fitting model');
tic;
md = fitcknn(features, labels, 'NumNeighbors', 4, 'Distance', 'spearman', 'NSMethod', 'exhaustive');
toc;

disp('Fit, running cross-validation');
tic;
cvmodel = crossval(md);
loss = kfoldLoss(cvmodel);
toc;
disp(loss);