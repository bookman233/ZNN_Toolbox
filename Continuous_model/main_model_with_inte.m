clc
clear
format long;

%% For complex-valued problems, x0 must be initialized as complex-valued number.
% x0 = [-1.1-1i; -1.2 + 1i; 2; 1; 0;0;0;0];
x0 = randn(4,1);
tspan = [0, 10];
iter_gap = 0.1;

%% Construct activation functions
AF = ["linear", "linear"];
AF_params_one = [2, 0.5, 2, 0.5];
AF_params_two = [2, 3, 2, 0.5];
AF_params = [AF_params_one; AF_params_two];

%% Define hyperparameters
gamma = 10;
mu = 10;
hyperparams = [gamma, mu];

%% Noise Define [noise type, strength] (noise - 0: Noise Free, 1: Constant, 2: Linear)
noise_info = [0, 0];

%% Model define
model = model_repo_with_inte;
ODE = ODE_Solvers;
options = odeset();
[t, x] = ode45(@model.NTZNNAF, tspan, x0, options, AF, AF_params, hyperparams, noise_info);
% [t, x] = ODE.RK4_Inte(@model.NTZNN, tspan, iter_gap, x0, AF, AF_params, hyperparams, noise_info);
% [t, x] = ODE.RK4_Inte(@model.NTGNN, tspan, iter_gap, x0, AF, AF_params, hyperparams, noise_info);

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

