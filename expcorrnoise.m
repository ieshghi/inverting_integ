function nums_out = expcorrnoise(n,corrn)

whitenoise = sqrt(2).*randn(n,1);
lambda = 1./corrn;
nums = zeros(n,1);
nums(1) = whitenoise(1);

for i = 2:n
    a = -lambda*nums(i-1);
    b = lambda;
    nums(i) = nums(i-1) + a + b*whitenoise(i);
end

nums_out = nums./sqrt(lambda);

end
