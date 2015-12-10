function [Y] = spectral_transform(data, K, sigma)
    n = size(data, 1);
    A = zeros(n, n);
    fprintf('Building affinity matrix... ');
    tic;
    for i = 1:n
        for j = 1:n
            if i == j
                A(i, j) = 0;
            else
                si = data(i, :);
                sj = data(j, :);
                A(i,j) = exp( - norm(si - sj)^2 / (2 * sigma^2) );
            end
        end
    end
    toc;
    fprintf('Done!\nBuilding D... ');
    tic;
    D = zeros(n, n);
    for i = 1:n
        D(i, i) = sum(A(i, :));
    end
    toc;
    fprintf('Done!\nBulding L... ');
    Dinv = inv(sqrt(D));
    L = Dinv * A * Dinv;
    fprintf('Done!\nFinding eivenvectors... ');
    [X, ~] = eigs(L,K);
    fprintf('Done!\n Normalizing... ');
    Y = X;
    for i = 1:n
        Y(i,:) = X(i,:) / norm(X(i, :), 2);
    end
end