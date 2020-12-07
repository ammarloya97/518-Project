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

%validationset
validationrange = round((length(onesindex)-percentage)/2)+percentage;
yvalidation = [y(onesindex(1,percentage+1:validationrange)) y(zerosindex(1,percentage+1:validationrange))];
xvalidation = [x(:,onesindex(1,percentage+1:validationrange)) x(:,zerosindex(1,percentage+1:validationrange))];

xnormalised = (xvalidation - means)./sigma;
xvalidation = xnormalised;

%testing set
ytesting = [y(onesindex(1,validationrange+1:end)) y(zerosindex(1,validationrange+1:end))];
xtesting = [x(:,onesindex(1,validationrange+1:end)) x(:,zerosindex(1,validationrange+1:end))];

xnormalised = (xtesting - means)./sigma;
xtesting = xnormalised;

%%
%classifier
lambda = linspace(0,0.005,10);
for k = 1:4
w0 = rand(size(x,1)+1,1);
 f = @(w)cross_entropy(w,xtraining,ytraining,lambda(k));

% train classifier
[W,fW] = gradient_descent(f,w0,0.1,1e-5,10000);


%creating weights mat
weightsmat(:,k) = W(:,end);

values(k) = length(find(abs(W(:,end))<0.001)); %dimensions that are reduced
costs(k) = fW(end);
%bar(W(:,end))

%testingseterror
xbar = [ones(1,length(ytraining));xtraining];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
errortesting(k) = 100- confusionmatrix(ytraining,yPredicted)*100;

%validation set error
xbar = [ones(1,length(yvalidation));xvalidation];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
errorvalidation(k) =100- confusionmatrix(yvalidation,yPredicted)*100;

xbar = [ones(1,length(ytesting));xtesting];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
accuracytestingmat(k) =100 - confusionmatrix(ytesting,yPredicted)*100;

end
clc
figure(1)
hold on
plot(lambda(1:k),errortesting(1:k),'ko--')
plot(lambda(1:k),errorvalidation(1:k),'go--')
plot(lambda(1:k),accuracytestingmat(1:k),'ro--')
title('Variation of error with lambda','FontSize',20)
xlabel('lambda','FontSize',16)
ylabel('error','FontSize',16)
legend('Training Set Error','Validation Set Error','Testing Set Error','FontSize',16)
%%

%organising results
xbar = [ones(1,length(ytraining));xtraining];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
accuracytraining = confusionmatrix(ytraining,yPredicted)*100;

%organising results
xbar = [ones(1,length(ytesting));xtesting];
W= W(:,end);
yPredicted = sigmoid(xbar'*W)';
i = find(yPredicted>0.5);
yPredicted(i)=1;
i = find(yPredicted<0.5);
yPredicted(i)=0;

%finding accuracy and confusion matrix
accuracytesting = confusionmatrix(ytesting,yPredicted)*100;
%plot(1:length(fW),fW)

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

function cost = cross_entropy(w,x,y,lambda)
% w is a vector
P = length(y);
xbar = [ones(1,P); x];
SIG = sigmoid(xbar'*w);
g = (1/P) * ...
    sum( ...
    - repmat(y',1,size(w,2)) .* log(SIG)...
    - (1-repmat(y',1,size(w,2))) .* log(1-SIG));
%h  = lambda*repmat(norm(abs(w(2:end,end))),1,size(w,2));
h = lambda*sum(abs(w(2:end,:)));
cost = g +h;
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