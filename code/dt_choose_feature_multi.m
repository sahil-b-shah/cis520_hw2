function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% YOUR CODE GOES HERE

% Get entropy of the Y distribution.
temp = mean(Z);
H = multi_entropy(mean(Z)');
% disp(mean(Z));
% disp(H);

% Compute conditional entropy for each feature.
ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

%Precompute values
num_of_k = size(Z, 2);

% Compute the IG of the best split with each feature. This is vectorized
% so that, for each feature, we compute the best split without a second for
% loop. Note that if we were guaranteed binary features, we could vectorize
% this entire loop with the same procedure.
t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', size(Z, 1));
for i = colidx
    t.timeleft();

    % Check for constant values.
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    % Compute up to 10 possible splits of the feature.
    r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), min(10, numel(Xrange{i})));
%     disp('r');
%     disp(r);
    split_f = bsxfun(@le, X(:,i), r(1:end-1));
%     disp('split');
%     disp(split_f);
    
    Z_proj = reshape(Z, size(Z, 1), 1, num_of_k);
%     disp('z proj');
%     disp(Z_proj);
    
    
    % Make Nx10xK matrix for each indicator variable
    k_given_x = bsxfun(@and, Z_proj, split_f);
    k_given_notx = bsxfun(@and, Z_proj, ~split_f);
%     disp('k given x');
%     disp(k_given_x);
    
%     disp('k given not x');
%     disp(k_given_notx);
    
    % Make a 1xSplitxK 
    total_k_given_x_per_split = sum(k_given_x);
    total_k_given_notx_per_split = sum(k_given_notx);
%     disp('hi');
%     disp(total_k_given_x_per_split);
    
    % Make into KxSplit matrix
    total_k_given_x_per_split = squeeze(total_k_given_x_per_split);
    total_k_given_notx_per_split = squeeze(total_k_given_notx_per_split);
    
    % If K!=1 transpose
    if (num_of_k ~= 1) && (size(split_f, 2) ~= 1)
        total_k_given_x_per_split = total_k_given_x_per_split';
        total_k_given_notx_per_split = total_k_given_notx_per_split';
    end
    
%     disp('hi1');
%     disp(total_k_given_x_per_split);
    
    % Get total numbers in split and replicate for number of labels K
    total_per_split = sum(split_f);
    total_per_not_split = sum(~split_f);
    diviser = repmat(total_per_split, [size(Z, 2), 1, 1]);
    diviser_not_split = repmat(total_per_not_split, [size(Z, 2), 1, 1]);
%     disp('diviser');
%     disp(diviser);
    p_per_k_given_split = total_k_given_x_per_split./diviser;
    p_per_k_given_not_split = total_k_given_notx_per_split./diviser_not_split;
    
%     disp('p_per_k_given_not_split');
%     disp(p_per_k_given_not_split);
    
%     disp('p_per_k_given_split');
%     disp(p_per_k_given_split);
    
    px = mean(split_f);
    cond_H = px.*multi_entropy(p_per_k_given_split) + ...
            (1-px).*multi_entropy(p_per_k_given_not_split);
%     disp('cond_H');
%     disp(cond_H);
    
    [ig(i) best_split] = max(H-cond_H);
    split_vals(i) = r(best_split);
end

% Choose feature with best split.
[max_ig fidx] = max(ig);
val = split_vals(fidx);