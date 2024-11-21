clc
clear
format long;

%% For complex-valued problems, x0 must be initialized as complex-valued number.
% x0 = [-1.1-1i; -1.2 + 1i; 2; 1]; % For OZNN and OGNN
x0 = [-1.1;-1.2];
% x0 = [-1.122511280687204;-1.236883085813983; 2; 1; 0;0;0;0];
gamma = 5;
tspan = [0, 2];
iter_gap = 0.01;

%% Construct activation functions
AF = 'hs';
hyper_params = [3, 0.5, 2, 0.5];

%% Noise Define (0: Noise Free, 1: Constant, 2: Linear, 3: Random)
noise_info = [1, 0];

%% Model define
model = model_repo;
ODE = ODE_Solvers;
options = odeset();
[t, x] = ode45(@model.OZNN, tspan, x0, options, AF, hyper_params, gamma, noise_info);
% [t, x] = ODE.RK4(@model.OZNN, tspan, iter_gap, x0, AF, hyper_params, gamma, noise_info);

%% Residual error compute
Mat_Vec = Matrix_Vec;
for j = 1:length(t)
    T = t(j);
    D = Mat_Vec.D(T);
    w = Mat_Vec.w(T);
    X = x(j,1:length(w));
    Err = D*X.'+w;
    nerr(j) = norm(Err);
end

%% Result print
% figure
set(gca,'FontSize',14)
plot(t, nerr, 'LineWidth', 2);
xlabel('{\itt} (s)')
ylabel('||{\itE}(t)||_F')
hold on;