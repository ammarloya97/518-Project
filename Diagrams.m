clc; clear all;
load('pressuredatacost.mat'); load('pressuredatanormalisedcost.mat'); load('pressuredataboosted.mat')

figure(1) 
hold on
plot(1:length(pressuredatacost),pressuredatacost)
plot(1:length(pressuredatanormalisedcost),pressuredatanormalisedcost)
xlabel('Iterations','FontSize',16)
ylabel('Cost','FontSize',16)
title('Pressure Data','FontSize',20)
legend('Pressure Data', 'Pressure Data Normalised','FontSize',16)
load('pressuredatarawcost.mat'); load('pressuredatarawnormalised.mat'); load('pressuredatarawboosted.mat')
figure(2) 
hold on
plot(1:length(pressuredatarawcost),pressuredatarawcost)
plot(1:length(pressuredatarawnormalised),pressuredatarawnormalised)
xlabel('Iterations','FontSize',16)
ylabel('Cost','FontSize',16)
title('Pressure Data Raw','FontSize',20)
legend('Pressure Data Raw', 'Pressure Data Raw Normalised','FontSize',16)
