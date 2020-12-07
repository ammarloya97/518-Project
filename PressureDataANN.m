clc; clear all; close all;

%this file applies the cross entropy function over pressure data without any
%feature engineering and selection

%pressuredata
%import file
pressuredata = importdata('pressure_data.xls');

%%
%sort data into x and y
y = pressuredata(:,1)';
x = pressuredata(:,2:end)';

%sorting data into testing and training sets
%testing set
onesindex = find(y==1);
zerosindex = find(y==0);
%softmax criterion
y(zerosindex) = -1;

%of data to be used as testing and training set
percentage = 70;
percentage = round(length(onesindex)*percentage/100);

%randomise training set
ytraining = [y(onesindex(1,1:percentage)) y(zerosindex(1,1:percentage))];
xtraining = [x(:,onesindex(1,1:percentage)) x(:,zerosindex(1,1:percentage))];
P = length(ytraining); %data set length

%reducing dimensions via PCA
Cov = (1/P)*xtraining*xtraining';
[V,D] = eig(Cov);
[D_sorted,id] = sort(diag(D),'descend');
A = D_sorted(D_sorted>0.0005);

% choose spanning set
Ctrain = [];
for i = 1:length(A)
    Ctrain = horzcat(Ctrain, V(:,id(i)));
    i = i + 1;
end

% project data  onto new spanning set
wtraining = linsolve(Ctrain'*Ctrain,Ctrain'*xtraining);

%standard normalising
means = mean(wtraining,2);
sigma = sqrt(1/P * sum((wtraining-means).^2,2));

wnormalised = (wtraining - means)./sigma;
wtraining = wnormalised;

%validationset
validationrange = round((length(onesindex)-percentage)/2)+percentage;
yvalidation = [y(onesindex(1,percentage+1:validationrange)) y(zerosindex(1,percentage+1:validationrange))];
xvalidation = [x(:,onesindex(1,percentage+1:validationrange)) x(:,zerosindex(1,percentage+1:validationrange))];

wvalidation = linsolve(Ctrain'*Ctrain,Ctrain'*xvalidation);

wnormalised = (wvalidation - means)./sigma;
wvalidation = wnormalised;

%testing set
ytesting = [y(onesindex(1,validationrange+1:end)) y(zerosindex(1,validationrange+1:end))];
xtesting = [x(:,onesindex(1,validationrange+1:end)) x(:,zerosindex(1,validationrange+1:end))];

wtesting = linsolve(Ctrain'*Ctrain,Ctrain'*xtesting);

wnormalised = (wtesting - means)./sigma;
wtesting = wnormalised;

%neurons
U=1:4:15;
% no. features
N=size(wtraining,1);

for k = 1:length(U)
% init. weights
w0 = randn((N+1)*U(k),1);
%run classifier
f = @(w)softmax(w,wtraining,ytraining,U(k));
[W,fW] = gradient_descent(f,w0,0.1,1e-5,2000);
W_opt=W(:,end);

%accuracy of training set
xbar = [wtraining];
yPredicted = sign(model(W_opt,xbar,U(k)));
error(k) = 100- confusionmatrix(ytraining,yPredicted)*100;

%accuracy of validation set
xbar = [wvalidation];
yPredicted = sign(model(W_opt,xbar,U(k)));
errorvalidation(k) = 100- confusionmatrix(yvalidation,yPredicted)*100;

figure(1)
hold on
plot(1:k,error(1:k),'ko--')
plot(1:k,errorvalidation(1:k),'go--')

%accuracy of testing set
xbar = [wtesting];
yPredicted = sign(model(W_opt,xbar,U(k)));
accuracy(k) = confusionmatrix(ytesting,yPredicted)*100;
end

%accuracy of testing set
xbar = [wtesting];
yPredicted = sign(model(W_opt,xbar,U(k)));
%accuracy = confusionmatrix(ytesting,yPredicted)*100

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
    i = find(yreal<1);
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
    Confusion = [a b; c d];
    accuracy = (a*0.5)/(a+c)+(d*0.5)/(d+b);

    
end
function a = model(w,x,U)
N = size(x,1);
W{1}=reshape(w(1:(N+1)*U),N+1,U);
xbar = [ones(1,size(x,2));x];
a = tanh(xbar'*W{1})';

a = sum(a,1);
end

function cost = softmax(w,x,y,U)
% w is a 2-D vector (or matrix with two rows)
P = length(y);
for i = 1 : size(w,2)
    wi=w(:,i);
    model_i = model(wi,x,U);
    cost(i) = (1/P) * ...
        sum( ...
        log( 1 + exp(-y.*model_i) ));
end
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