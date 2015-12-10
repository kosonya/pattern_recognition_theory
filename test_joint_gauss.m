function [scores] = test_joint_gauss(features, mus, sigmas, posprior)
    scores = zeros(size(features, 1), 2);
    scores(:,1) = mvnpdf(features, mus{1}, sigmas{1}) * (1 - posprior);
    scores(:,2) = mvnpdf(features, mus{2}, sigmas{2}) * posprior;
    scores = bsxfun(@rdivide, scores, sum(scores, 2));
end