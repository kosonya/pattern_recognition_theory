curfile = mfilename('fullpath');
curfile_split = strsplit(curfile, filesep);
root_folder = curfile_split(1:end-2);

n_eigs = 20;

XY = {'X', 'Y'};
trval = {'Train', 'validation'};
for i = 1:numel(XY)
    for j = 1:numel(trval) 
        mat_fname = strjoin([root_folder, sprintf('%s_%s.mat', XY{i}, trval{j})], filesep);
        fprintf('loading %s\n', mat_fname);
        tic;
        load(mat_fname);
        toc;
    end
end

fprintf('Subtracting means... ');
X_train_means = mean(X_train);
X_train = bsxfun(@minus, X_train, X_train_means);
X_Validation = bsxfun(@minus, X_Validation, X_train_means);

fprintf('Training independent Gauss on raw data... ');
tic;
[mus, sigmas, posprior] = train_independent_gauss(X_train, Y_train);
fprintf('Done! ');
toc;

fprintf('Making training set predictions... ');
tic;
[scores] = test_independent_gauss(X_train, mus, sigmas, posprior);
fprintf('Done! ');
toc;

fprintf('Building ROC... ');
tic;
[train_X, train_Y, ~, train_AUC] = perfcurve(Y_train, scores(:,2), 1);
fprintf('Done! ');
toc;

fprintf('Making test set predictions... ');
tic;
[scores] = test_independent_gauss(X_Validation, mus, sigmas, posprior);
fprintf('Done! ');
toc;

fprintf('Building ROC... ');
tic;
[test_X, test_Y, ~, test_AUC] = perfcurve(Y_Validation, scores(:,2), 1);
fprintf('Done! ');
toc;


figure('name', 'Raw data set independent Gauss');
plot(train_X, train_Y);
hold;
plot(test_X, test_Y);
xlabel('False positive rate')
ylabel('True positive rate')
title('Raw data set independent Gauss ROC');
legend(sprintf('Training set AUC = %f', train_AUC), sprintf('Test set AUC = %f', test_AUC), 'Location', 'southeast');

drawnow;

fprintf('Computing eigenvectors for training data... ');
tic;
[eigenvectors,~,eigenvalues] = pca(X_train);
projmat_full =  eigenvectors*inv(eigenvectors'*eigenvectors)*eigenvectors;
fprintf('Done! ');
toc;

n_eigses = {2, 8, 16, 32, 64, 128};
train_ROCs = cell(numel(n_eigses), 5);
test_ROCs = cell(size(train_ROCs));

for i = 1:length(n_eigses)
    n_eigs = n_eigses{i};
    fprintf('Traing first %d principal components\n', n_eigs);
    projmat = projmat_full(:,1:n_eigs);

    fprintf('Projecting data... ');
    tic;
    X_train_pca = X_train * projmat;
    X_Validation_pca = X_Validation * projmat;
    fprintf('Done! ');
    toc;
      
    fprintf('Training independent Gauss for n_eigs = %d\n', n_eigs);
    tic;
    [mus, sigmas, posprior] = train_independent_gauss(X_train_pca, Y_train);
    fprintf('Done! ');
    toc;
    fprintf('Making training set predictions... ');
    tic;
    [train_scores] = test_independent_gauss(X_train_pca, mus, sigmas, posprior);
    fprintf('Done! ');
    toc;
    fprintf('Building training ROC... ');
    tic;
    [X, Y, ~, AUC] = perfcurve(Y_train, train_scores(:,2), 1);
    fprintf('Done! ');
    toc;
    train_ROCs{i, 1} = X;
    train_ROCs{i, 2} = Y;
    train_ROCs{i, 3} = AUC;
    train_ROCs{i, 4} = n_eigs;
    train_ROCs{i, 5} = sprintf('n_eigs = %d, AUC = %f', n_eigs, AUC);
    
    fprintf('Making test set predictions... ');
    tic;
    [test_scores] = test_independent_gauss(X_Validation_pca, mus, sigmas, posprior);
    fprintf('Done! ');
    toc;
    fprintf('Building validation ROC... ');
    tic;
    [X, Y, ~, AUC] = perfcurve(Y_Validation, test_scores(:,2), 1);
    fprintf('Done! ');
    toc;
    test_ROCs{i, 1} = X;
    test_ROCs{i, 2} = Y;
    test_ROCs{i, 3} = AUC;
    test_ROCs{i, 4} = n_eigs;
    test_ROCs{i, 5} = sprintf('n_eigs = %d, AUC = %f', n_eigs, AUC);
end

figure('name', 'train Gauss ROCs');
for i=1:size(train_ROCs, 1)
    hold;
    plot(train_ROCs{i, 1}, train_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Training set ROCs for independent Gauss');
legend({train_ROCs{:,5}}, 'Location', 'southeast');

figure('name', 'test Gauss ROCs');
for i=1:size(test_ROCs, 1)
    hold;
    plot(test_ROCs{i, 1}, test_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Test set ROCs for independent Gauss');
legend({test_ROCs{:,5}}, 'Location', 'southeast');



