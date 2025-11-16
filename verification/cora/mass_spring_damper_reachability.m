function res = mass_spring_damper_reachability()
close all; clc;

% Plotting preamble
set(0,'DefaultFigureWindowStyle','docked')
font=12;
set(groot, 'defaultAxesTickLabelInterpreter', 'latex'); 
set(groot, 'defaultLegendInterpreter', 'latex');
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesFontSize', font)
set(0, 'defaultLegendFontSize', font)
set(0, 'defaultAxesFontName', 'Times New Roman');
set(0, 'defaultLegendFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 0.5);

% System parameters
m = 1.0;           % Mass
b = 5;             % Damping constant
k = 5;             % Spring constant
y_0 = -10;         % Initial position

% Plot from initial point with ode45
% sys_dyn = @(t, y) [y(2); (-b/m) * y(2) - (k/m) * (y(1))];
% [t, y] = ode45(sys_dyn, [0 10], [y_0; 0]);
% f = figure; tiledlayout(1,3); nexttile;
% plot(t, y(:, 1),'k'); xlabel('$t$'); ylabel('$y_1$'); title("$y_1$ (position) vs time"); nexttile;
% plot(t, y(:, 2),'k'); xlabel('$t$'); ylabel('$y_2$'); title("$y_2$ (velocity) vs time"); nexttile;
% plot(y(:, 1), y(:, 2),'k'); xlabel('$y_1$'); title("$y_1$ vs $y_2$"); ylabel('$y_2$');

% System equations (for CORA)
A = [0 1; -k/m -b/m]; %where y1 = x, y2 = x_dot\\
% first ODE: y1_dot = y2; second ODE: y2_dot = -b/m * y2 -k/m *y1
B = [0; 0]; %coefficients of inputs (in this case, there is no input forcing to the system)
sys = linearSys(A,B);

% Reachability settings
params.tFinal = 6;
options.timeStep = 0.01; 
options.taylorTerms = 4; %default
options.zonotopeOrder = 20; %default

% Verification specifications
params.R0 = zonotope(interval([y_0 - 2.5;0],[y_0 + 2.5;5])); % initial reachable \\
% set defined using zonotopes
goalSet = interval([-0.1; -0.1],[0.1; 0.1]);
avoidSet = interval([0; 10],[5; 15]);
spec1 = specification(goalSet, 'safeSet', interval(params.tFinal));
spec2 = specification(avoidSet, 'unsafeSet', interval(0,params.tFinal));
spec = add(spec1,spec2);

% Verification
tic;
[R,res] = reach(sys, params, options, spec);
toc;

if res == true
    disp('Result: Verification success')
else
    disp('Result: Verification failure')
end

% Simulation
simOpt.points = 25;
simOpt.type = 'gaussian';
simRes = simulateRandom(sys, params, simOpt);

% Plotting -----------------------------------------------
disp("Plotting..")
f = figure;
dims = [1 2];
set(0,'DefaultFigureWindowStyle','docked')
tiledlayout(1,2);
spec = specification(goalSet, 'safeSet', interval(0, params.tFinal));

% Plots 1 and 2: position and velocity vs time
for k = 1:2
    nexttile; hold on; box on;  grid on;

    % Plot goal set
    plotOverTime(spec, k, 'DisplayName', 'Goal set');
    
    if k == 2
        title("$y_2$ (velocity) vs time");
    else
        title("$y_1$ (position) vs time");
    end
    
    % Plot reachable set
    useCORAcolors('CORA:contDynamics')
    plotOverTime(R, k, 'DisplayName', 'Reachable set');
    
    % Plot initial set
    plotOverTime(R(1).R0, k, 'DisplayName', 'Initial set');
    
    % Plot simulations
    plotOverTime(simRes, k, 'DisplayName', 'Simulations')
    xlim([0 params.tFinal]) %max(simRes(1).t(1))
    xlabel('$t$');
    ylabel(['$y_{',num2str(k),'}$']);

end

% Plot 3: position vs velocity
f2 = figure;
tiledlayout(1,3); nexttile;
hold on; box on; grid on;
plot(spec, dims,'DisplayName', 'Goal set');
plot(spec2, dims,'DisplayName', 'Unsafe set');
useCORAcolors("CORA:contDynamics")
plot(R,dims,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',5);
plot(R(1).R0,dims, 'DisplayName','Initial set');
plot(simRes,dims,'DisplayName','Simulations');
xlabel(['$y_{',num2str(dims(1)),'}$']);
ylabel(['$y_{',num2str(dims(2)),'}$']);
title("$y_2$ vs $y_1$")
daspect([1 1 1])
lgd = legend;
lgd.Layout.Tile = 3;

% Plot 4: final reachable set of position vs velocity
nexttile
hold on; box on; grid on;
plot(spec, dims,'DisplayName', 'Goal set');
useCORAcolors("CORA:contDynamics")
plot(R.timePoint.set{length(R.timePoint.set), 1},dims,'DisplayName', 'Reachable set','FaceAlpha',1)
axis manual
plot(R(1).R0,dims, 'DisplayName','Initial set');
xlabel(['$y_{',num2str(dims(1)),'}$']);
ylabel(['$y_{',num2str(dims(2)),'}$']);
title("$y_2$ vs $y_1$ at t=" + num2str(params.tFinal))
daspect([1 1 1])

% exportgraphics(f2,'mass_spring_damper_reachability.png','Resolution',600)
end