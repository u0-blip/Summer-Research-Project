%%
freq = [1.5, 2.26, 3, 3.7];
res = [1.07487079014246e-06,5.41462462904420e-07,2.72407044060959e-07,1.43851082514600e-07];
p = polyfit(freq, res, 2);

x1 = linspace(1, 6, 20);
% fitted_line = polyval(p,x1);

g = 'exp1';
f0 = fit(freq', res', g);

set(gcf,'color','w');

plot(freq, res, 'bo')
hold on
plot(x1, f0(x1))

ax = gca;
ax.TitleFontSizeMultiplier = 3;
legend('data points', 'exp fitted line')
title('frequency vs energy delivered')
xlabel('Frequency (GHz)')
ylabel('mean RMS value of E field strength')