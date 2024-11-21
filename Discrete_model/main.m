%% Code implementation of 
%% "Complex-Valued Discrete-Time Neural Dynamics for Perturbed Time-Dependent Complex Quadratic Programming With Applications"
%% Equ. 14

clear;
clc;
close all;

m = 7;   
tau = 0.0001;
tf=10;

% 7*1
z = rand(m,1)+i*(rand(m,1));
t = 0:tau:tf;

S = MatrixA(t(1)) ;
q = Vectorb(t(1));
z = z - pinv(S)*(S*z-q);

Fnorm = zeros(length(t)-1,1);
zdata = zeros(length(t)-1,1);

inte = zeros(7, 1);
for k = 2 : length(t)
    TTprev = t(k-1);
    Sprev = MatrixA(TTprev);
    qprev = Vectorb(TTprev);
    
    TT = t(k);
    S = MatrixA(TT);
    q = Vectorb(TT);
    
    z = z + pinv(S) * (-(S-Sprev)*z + (q-qprev) - (S*z-q) - inte);
    err = S*z-q;
    inte = inte + err;
    
    zdata(k) = z(7);
    Fnorm(k) = norm(S*z-q);
    k
end

figure(1)
plot(Fnorm(2:end), 'linewidth',2);
hold on;

figure(2)
plot3(real(zdata), imag(zdata),t);
