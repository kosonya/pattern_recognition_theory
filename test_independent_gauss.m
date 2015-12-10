function [scores] = test_independent_gauss(features, mus, sigmas, posprior)
    scores = zeros(size(features, 1), 2);
    pos_pdfs = zeros(size(features));
    neg_pdfs = zeros(size(features));
    for curcol = 1:size(features, 2)
        neg_pdfs(:,curcol) = normpdf(features(:,curcol), mus(1, curcol), sigmas(1, curcol));
        pos_pdfs(:,curcol) = normpdf(features(:,curcol), mus(2, curcol), sigmas(2, curcol));
    end
    scores(:,1) = prod(neg_pdfs, 2) * (1 - posprior);
    scores(:,2) = prod(pos_pdfs, 2) * posprior;
    scores = bsxfun(@rdivide, scores, sum(scores, 2));
end