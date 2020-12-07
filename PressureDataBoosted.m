clc; clear all; close all;

%this file applies the cross entropy function over pressure data without any
%feature engineering and selection

%pressuredata
%import file
pressuredata = importdata('pressure_data_raw.xls');
%%
%sort data into x and y
y = pressuredata(:,1)';
x = pressuredata(:,2:end)';
n = size(x,1); %dimension

%sorting data into testing and training sets
%testing set
onesindex = find(y==1);
zerosindex = find(y==0);

%of data to be used as testing and training set
percentage = 70;
percentage = round(length(onesindex)*percentage/100);

%randomise training set
ytraining = [y(onesindex(1,1:percentage)) y(zerosindex(1,1:percentage))];
xtraining = [x(:,onesindex(1,1:percentage)) x(:,zerosindex(1,1:percentage))];
P = length(ytraining); %data set length

%standard normalising
means = mean(xtraining,2);
sigma = sqrt(1/P * sum((xtraining-means).^2,2));

xnormalised = (xtraining - means)./sigma;
xtraining = xnormalised;

%testing set
ytesting = [y(onesindex(1,percentage+1:end)) y(zerosindex(1,percentage+1:end))];
xtesting = [x(:,onesindex(1,percentage+1:end)) x(:,zerosindex(1,percentage+1:end))];

xnormalised = (xtesting - means)./sigma;
xtesting = xnormalised;

%%
%classifier
xbar = [ones(1,length(ytraining)); xtraining];
wsetxvector = [ones(1,length(ytraining))];
vectorsdone = 0;

f = @(w)cross_entropy(w,xbar(1,:),ytraining,0);
w0 = rand(1);
[W,fW] = gradient_descent(f,w0,0.1,1e-6,100000);
weights=W(end);
cost = fW(end);

for i=1:n
    for j = 1:n
        w0=rand(1);
        if ismember(j,vectorsdone)==1
        sweight(j) = 1000;
        scost(j) = 1000;
            continue
        end
        f = @(w)cross_entropy(w,xtraining(j,:),ytraining,wsetxvector'*weights');
        [W,fW] = gradient_descent(f,w0,0.1,1e-6,100000);
        sweight(j) = W(end);
        scost(j) = fW(end);
    end
    [cost(i+1),index] = min(scost);
    weights(i+1) = sweight(index);
    wsetxvector(i+1,:) = xtraining(index,:);
    vectorsdone(i) = index;
    sweight = 0;
    scost = 0;
    if cost(i)-cost(i+1)<1e-5
        break
    end
end
W=weights';


%%
%organising results
xbar = [ones(1,length(ytesting)); xtesting(vectorsdone,:)];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
accuracytesting = confusionmatrix(ytesting,yPredicted)*100;

%organising results
xbar = [ones(1,length(ytraining)); xtraining(vectorsdone,:)];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
accuracytraining = confusionmatrix(ytraining,yPredicted)*100;

plot(1:length(fW),fW)
figure(2)
plot(1:length(cost),cost)
ylabel('Cost','FontSize',16)
xlabel('Dimensions','FontSize',16)
title('Variation of cost with dimensions added','FontSize',20)
function accuracy = confusionmatrix(yreal,ypredicted)
    i = find(yreal==1);
    A1 = 0;
    a =0; b= 0; c=0; d=0;
    for j = 1:length(i)
    if yreal(i(j))==ypredicted(i(j))
        error = 0;
        a = a+1;
    else 
        error = 1;
        b = b+1;
    end
    A1 = A1+error;
    end
    
    A1 = 1 - A1/length(i);
    i = find(yreal==0);
    A0 = 0;
    for j = 1:length(i)
        if yreal(i(j))==ypredicted(i(j))
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
    Confusion = [a b; c d]
    accuracy = (a*0.5)/(a+c)+(d*0.5)/(d+b);

    
end


function cost = cross_entropy(w,x,y,wset)
% w is a vector
P = length(y);
xbar = x;
SIG = sigmoid((xbar'*w)+wset);
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
function [W,fW] = gradient_descent(f,w0,alpha,error,iters)
 
k = 2;
W(:,k) = w0;
fW(k) = f(w0);
 
while abs(fW(k)-fW(k-1)) > error
 
    if abs(fW(k)-fW(k-1))<1
        alpha = 0.01;
    end
    if abs(fW(k)-fW(k-1))<0.1
        alpha = 0.001; %reduce stepsize to obtain accurate minimum and avoid divergence/oscillation
    end
    grad = approx_grad(f,W(:,k),.000001);
    W(:,k+1) = W(:,k) - alpha*(grad')/norm(grad);
    fW(k+1) = f(W(:,k+1));
    k = k+1;
    if k>iters
        break
    end
end
 
end