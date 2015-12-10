curfile = mfilename('fullpath');
curfile_split = strsplit(curfile, filesep);
root_folder = curfile_split(1:end-2);

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

X_train = X_train(1:20000, :);
Y_train = Y_train(1:20000, :);
X_Validation = X_Validation(1:2000, :);
Y_Validation = Y_Validation(1:2000, :);

fprintf('Subtracting means... ');
X_train_means = mean(X_train);
X_train = bsxfun(@minus, X_train, X_train_means);
X_Validation = bsxfun(@minus, X_Validation, X_train_means);

fprintf('Doing spectral stuff...\n ');
tic;
X_combined = [X_train; X_Validation];
X_combined = spectral_transform(X_combined, 2, 1);
X_train = X_combined(1:size(X_train, 1), :);
X_Validation = X_combined( (size(X_train, 1) + 1):end, :);
fprintf('Done with spectral! ');
toc;


Ks = {1, 2, 4, 8, 16, 32, 64};
models = cell(size(Ks));
train_ROCs = cell(numel(Ks), 5);
test_ROCs = cell(size(train_ROCs));
for i = 1:length(Ks)
    k = Ks{i};
    fprintf('Training knn classifier with k = %d\n', k);
    md = fitcknn(X_train, Y_train, 'NumNeighbors', k);
    fprintf('Done! ');
    toc;
    models{i} = md;
    fprintf('Making training set predictions... ');
    tic;
    [~, train_scores] = predict(md, X_train);
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
    [~, test_scores] = predict(md, X_Validation);
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

figure('name', 'train spectral ROCs');
for i=1:size(train_ROCs, 1)
    hold;
    plot(train_ROCs{i, 1}, train_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Training set spectral ROC');
legend({train_ROCs{:,5}}, 'Location', 'southeast');

figure('name', 'test spectral ROCs');
for i=1:size(test_ROCs, 1)
    hold;
    plot(test_ROCs{i, 1}, test_ROCs{i, 2});
    hold;
end
xlabel('False positive rate')
ylabel('True positive rate')
title('Test set spectral ROC');
legend({test_ROCs{:,5}}, 'Location', 'southeast');

