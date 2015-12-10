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

fprintf('Computing eigenvectors for training data... ');
tic;
[eigenvectors,~,eigenvalues] = pca(X_train);
fprintf('Done! ');
toc;

projmat_full =  eigenvectors*inv(eigenvectors'*eigenvectors)*eigenvectors;
projmat = projmat_full(:,1:n_eigs);

fprintf('Projecting data... ');
tic;
X_train_pca = X_train * projmat;
X_Validation_pca = X_Validation * projmat;
fprintf('Done! ');
toc;

Ks = {1, 2, 4, 8, 16, 32, 64};
models = cell(size(Ks));
train_ROCs = cell(numel(Ks), 5);
test_ROCs = cell(size(train_ROCs));
for i = 1:length(Ks)
    k = Ks{i};
    fprintf('Training knn classifier with k = %d\n', k);
    md = fitcknn(X_train_pca, Y_train, 'NumNeighbors', k, 'NSMethod', 'exhaustive', 'Distance', 'cosine');
    fprintf('Done! ');
    toc;
    models{i} = md;
    fprintf('Making training set predictions... ');
    tic;
    [~, train_scores] = predict(md, X_train_pca);
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
    train_ROCs{i, 4} = k;
    train_ROCs{i, 5} = sprintf('k = %d, AUC = %f', k, AUC);
    
    fprintf('Making test set predictions... ');
    tic;
    [~, test_scores] = predict(md, X_Validation_pca);
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
    test_ROCs{i, 4} = k;
    test_ROCs{i, 5} = sprintf('k = %d, AUC = %f', k, AUC);
    
end

figure('name', 'train n_eig = 20 ROCs');
for i=1:size(train_ROCs, 1)
    hold;
    plot(train_ROCs{i, 1}, train_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Training set ROC for n\_eig = 20');
legend({train_ROCs{:,5}}, 'Location', 'southeast');

figure('name', 'test n_eig = 20 ROCs');
for i=1:size(test_ROCs, 1)
    hold;
    plot(test_ROCs{i, 1}, test_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Test set ROC for n\_eig = 20');
legend({test_ROCs{:,5}}, 'Location', 'southeast');

drawnow;

n_eigses = {2, 8, 16, 32, 64, 128};
models = cell(size(n_eigses));
train_ROCs = cell(numel(n_eigses), 5);
test_ROCs = cell(size(train_ROCs));

k = 20;

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
      
    fprintf('Training knn classifier with k = %d, n_eigs = %d\n', k, n_eigs);
    md = fitcknn(X_train_pca, Y_train, 'NumNeighbors', k, 'NSMethod', 'exhaustive', 'Distance', 'cosine');
    fprintf('Done! ');
    toc;
    models{i} = md;
    fprintf('Making training set predictions... ');
    tic;
    [~, train_scores] = predict(md, X_train_pca);
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
    [~, test_scores] = predict(md, X_Validation_pca);
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

figure('name', 'train k = 20 ROCs');
for i=1:size(train_ROCs, 1)
    hold;
    plot(train_ROCs{i, 1}, train_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Training set ROC for k = 20');
legend({train_ROCs{:,5}}, 'Location', 'southeast');

figure('name', 'test k = 15 ROCs');
for i=1:size(test_ROCs, 1)
    hold;
    plot(test_ROCs{i, 1}, test_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Test set ROC for k = 20');
legend({test_ROCs{:,5}}, 'Location', 'southeast');