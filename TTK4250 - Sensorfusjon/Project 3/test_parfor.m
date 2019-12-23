N = 100000000;
a = ones(N,1);
b = ones(N,1);
c = ones(N,1);
tic
for i = 1:N
    c(i) = a(i) + b(i);
end
t1 = toc;

a = ones(N,1);
b = ones(N,1);
c = ones(N,1);
tic
parfor i = 1:N
    c(i) = a(i) + b(i);
end
t2 = toc;

fprintf("Time for loop: %f\nTime parfor loop: %f\n", t1, t2);