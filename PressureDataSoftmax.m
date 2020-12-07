clc; clear all; close all;
%pressuredata
%import file
pressuredata = importdata('pressure_data.xls');
%%
%sort data into x and y
y = pressuredata(:,1)';
x = pressuredata(:,2:end)';
P = length(y); %data set length

ysoftmax = pressuredata(:,1)';
i= find(ysoftmax==0);
ysoftmax(i)=-1;

%sorting data into testing and training sets
%testing set

%training set

%PCA 
Means = mean(x,2);
X = x- Means;
Cov = (1/P)*X*X';
[V,D] = eig(Cov);
[D_sorted,id] = sort(real(diag(D)),'descend');
V = real(V);

%choosing a spanning set in nd
dimensions = 20;
C = [V(:,id(1):id(dimensions))];
%w = linsolve(C'*C,C'*X);
%xp = C*w;

%%
%classifier
w0 = rand(size(x,1)+1,1);

f = @(w)soft_max(w,x,ysoftmax);

% train classifier
[W,fW] = gradient_descent(f,w0,0.01,2000);

%%
%organising results
xbar = [ones(1,P); x];
W= W(:,end);
%yPredicted = sigmoid(xbar'*W);

yPredicted = tanh(xbar'*W)';

% i = find(yPredicted>0.5);
% yPredicted(i)=1;
% i = find(yPredicted<0.5);
% yPredicted(i)=0;

i = find(yPredicted>0);
yPredicted(i)=1;
i = find(yPredicted<0);
yPredicted(i)=-1;

%finding accuracy and confusion matrix

y=ysoftmax;
i = find(y==1);
A1 = 0;
a =0; b= 0; c=0; d=0;

for j = 1:length(i)
    if y(i(j))==yPredicted(i(j))
        error = 0;
        a = a+1;
    else 
        error = 1;
        b = b+1;
    end
    A1 = A1+error;
end
A1 = 1 - A1/length(i);
% i = find(y==0);
i = find(y==-1);
A0 = 0;
for j = 1:length(i)
    if y(i(j))==yPredicted(i(j))
        error = 0;
        d = d+1;
    else 
        error = 1;
        c=c+1;
    end
    A0 = A0+error;
end

A0 = 1 - A0/length(i);
Abal = (A0+A1)/2;
Confusion = [a b; c d];
Abal1 = (a*0.5)/(a+c)+(d*0.5)/(d+b)

function cost = soft_max(w,x,y)
% w is a 3-D vector (or matrix with three rows)
P = length(y);
xbar = [ones(1,P); x];
cost = (1/P) * ...
    sum( ...
    log(1+exp(-repmat(y',1,size(w,2)).*xbar'*w)));
end

function cost = cross_entropy(w,x,y)
% w is a vector
P = length(y);
xbar = [ones(1,P); x];
SIG = sigmoid(xbar'*w);
cost = (1/P) * ...
    sum( ...
    - repmat(y',1,size(w,2)) .* log(SIG)...
    - (1-repmat(y',1,size(w,2))) .* log(1-SIG));
end

function s = sigmoid(x)
s = 1./(1+exp(-x));
s(s==1) = .9999;
s(s==0) = .0001;
end

function grad = approx_grad(f,w0,delta)
N = length(w0);
dw = delta*eye(N);
grad = ( f(w0+dw) - f(w0) )/delta;
end

function [W,fW] = gradient_descent(f,w0,alpha,n_iter)

k = 1;
W = w0;
fW = f(w0);

while k < n_iter
    grad = approx_grad(f,W(:,k),.0001);
    W(:,k+1) = W(:,k) - alpha*(grad')/norm(grad);
    fW(k+1) = f(W(:,k+1));
    k = k+1;
end

end