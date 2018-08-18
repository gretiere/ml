function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% 8 values for both C and sigma = 8^2 combinations
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for i=1:length(C_vec)
    C_try = C_vec(i);
    for j=1:length(sigma_vec)
        % some code
        fprintf('Valeurs de i = %f et j = %f.\n', i, j);      
        sigma_try = sigma_vec(j);
        % on entraîne avec chaque combinaison de C et sigma
        model = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try)); 
        % on récupère l'erreur de la prédiction sur les cross val datas
        predictions = svmPredict(model,Xval);
        error(j) = mean(double(predictions ~= yval));

        % si iter 1 alors on initialise le vecteur erreur sinon on append
        if (j==1)
            error_j = error(j);
        else
            error_j = [error_j ; error(j)];
        end

    end
    %some code
    if (i==1)
        error_i = error_j;
    else
        error_i = [error_i ; error_j];
    end    

end

error_i

% on recherche la valeur minimale dans le vecteur 
[r,c] = find(error_i==min(min(error_i)));

C = C_vec(c);
sigma = sigma_vec(r);


% =========================================================================

end
