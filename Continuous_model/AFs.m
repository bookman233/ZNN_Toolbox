function output = AFs
    output.linear = @Linear;
    output.powerQ = @Power_Q;
    output.power  = @PowerSum;
    output.PTC    = @PTC;
    output.hs     = @Hyperbolic_Sine;
    output.ps     = @Power_Sigmoid;
    output.bs     = @Bipolar_sigmoid;
    output.sbp    = @Sign_Bi_Power;
    output.wsbp   = @Weighted_Sign_Bi_Power;
    output.VAF    = @Versatile_AF;
    output.TAF    = @Tunable_AF;
    output.bound  = @Bound;
    output.ball   = @Ball;
    output.tp     = @Tanh_Power;
    output.BiP    = @Bi_Power;
    output.tl     = @Tanh_Linear;
end

% RNN Models for Dynamic Matrix Inversion: A Control-Theoretical Perspective
function out = Linear(E)
    for i = 1:length(E)
        out(i) = E(i);
    end
    out = out';
end

% Saturation-Allowed Neural Dynamics Applied to Perturbed Time-Dependent System of Linear Equations and Robots
function out = PowerSum(E)
    for i = 1:length(E)
        out(i) = E(i) + (E(i))^3 + (E(i))^5;
    end
    out = out';
end

% Finite-Time and Predefined-Time Convergence Design for Zeroing Neural Network: Theorem, Method, and Verification
% Section 4-B
function out = Power_Q(E, a, q)
    assert(a > 0, 'Parameter a is not in the feasible range (a > 0).')
    assert(q > 0, 'Parameter q is not in the feasible range (q > 1).')
    for i = 1:length(E)
        out(i) = a*(E(i).^q);
    end
    out = out';
end

% Finite-Time and Predefined-Time Convergence Design for Zeroing Neural Network: Theorem, Method, and Verification
% Section 4-B
function out = Tanh_Linear(E, a, xi)
    assert(a > 0, 'Parameter a is not in the feasible range (a > 0).')
    assert(xi > 0, 'Parameter xi is not in the feasible range (xi > 0).')
    for i = 1:length(E)
        out(i) = a*tanh(xi*E(i));
    end
    out = out';
end

% RNN Models for Dynamic Matrix Inversion: A Control-Theoretical Perspective
function out = Hyperbolic_Sine(E, zeta)
    assert(zeta >= 1, 'Parameter zeta is not in the feasible range (zeta >= 1).')
    for i = 1 : length(E)
        out(i) = (exp(zeta*E(i)) - exp(-zeta*E(i)));
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P15)
function out = Power_Sigmoid(E, zeta, m)
    assert(zeta >= 2, 'Parameter zeta is not in the feasible range (zeta >= 2).')
    assert((mod(m, 2) ~= 0) && m >= 3, 'Feasible range of parameter m >= 2 and m must be odd number.')
    for i = 1 : length(E)
        if abs(E(i)) < 1
            out(i) = ((1+exp(-zeta))/(1-exp(-zeta))) * ((1-exp(-zeta*E(i)))/(1+exp(-zeta*E(i))));
        else
            out(i) = (E(i))^m;
        end
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P15)
function out = Bipolar_sigmoid(E, zeta)
    assert(zeta >= 2, 'Parameter eta is not in the feasible range (zeta >= 2).')
    for i = 1 : length(E)
        out(i) = (1-exp(-zeta*E(i)))/(1+exp(-zeta*E(i)));
    end
    out = out';
end

% A Strictly Predefined-Time Convergent Neural Solution to Equality- and Inequality-Constrained Time-Variant Quadratic Programming
function out = PTC(E, t, Conv_time)
    assert(Conv_time > 0, 'Parameter Conv_time is not in the feasible range (Conv_time > 0).')
    for i = 1 : length(E)
        if t < Conv_time
            out(i) = (exp(E(i))-1)./((Conv_time-t).*exp(E(i)));
        else
            out(i) = E(i);
        end
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P42)
function out = Sign_Bi_Power(E, r)
    assert(r > 0 && r < 1, 'Parameter eta is not in the feasible range (0 < eta < 1).')
    for i = 1 : length(E)
        out(i) = 0.5*(abs(E(i))^r + abs(E(i))^(1/r)) * sign(E(i));
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P174)
function out = Weighted_Sign_Bi_Power(E, r, k1, k2, k3)
    assert(r > 0 && r < 1, 'Parameter eta is not in the feasible range (0 < eta < 1).')
    assert((k1>0 && k1<1) && (k2>0 && k2<1) && (k3>0 && k3<1), 'Parameter k is not in the feasible range (0 < k < 1).')
    for i = 1 : length(E)
        out(i) = 0.5*(k1*abs(E(i))^r + k2*abs(E(i))^(1/r)) * sign(E(i)) + 0.5*k3*E(i);
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P42)
function out = Versatile_AF(E, eta, w, a1, a2, a3, a4)
    assert(eta > 0 && eta < 1, 'Parameter eta is not in the feasible range (0 < eta < 1).')
    assert(w > 1, 'Parameter w is not in the feasible range (w > 1).')
    assert(a1>0 && a2>0 && a3>=0 && a4>=0, 'Parameter a is not in the feasible range (a1 and a2 > 0, a3 and a4 >= 0).')
    for i = 1 : length(E)
        out(i) = (a1*abs(E(i))^eta + a2*abs(E(i))^w) * sign(E(i)) + a3*E(i) + a4*sign(E(i));
    end
    out = out';
end

% Reference: Zeroing Neural Networks: Finite-time Convergence Design,
% Analysis and Applications(P63)
function out = Tunable_AF(E, eta, k1, k2, k3)
    assert(eta > 0 && eta < 1, 'Parameter eta is not in the feasible range (0 < eta < 1).')
    assert(k1 > 0 && k2 > 0 && k3 > 0, 'Parameter k is not in the feasible range (k > 0).')
    for i = 1 : length(E)
        out(i) = (k1*abs(E(i))^eta + k2*abs(E(i))^(1/eta)) * sign(E(i)) + k3*E(i);
    end
    out = out';
end

% Reference: RNN for Solving Time-Variant Generalized Sylvester Equation With Applications to Robots and Acoustic Source Localization
function out = Bound(E, b)
    assert(b > 0, 'Parameter is not in the feasible range (> 0).')
    for i = 1 : length(E)
        if E(i) > b
            out(i) = b;
        elseif E(i) < -b
            out(i) = -b;
        else
            out(i) = E(i);
        end
    end
    out = out';
end

% Reference: RNN for Solving Time-Variant Generalized Sylvester Equation With Applications to Robots and Acoustic Source Localization
function out = Ball(E, b)
    assert(b > 0, 'Parameter b is not in the feasible range (> 0).')
    for i = 1 : length(E)
        if norm(E) > b
            out(i) = b*(E(i)/norm(E));
        else
            out(i) = E(i);
        end
    end
    out = out';
end

% Finite-Time and Predefined-Time Convergence Design for Zeroing Neural Network: Theorem, Method, and Verification
% Section 4-C
function out = Tanh_Power(E, a, p)
    assert(a > 0, 'Parameter a is not in the feasible range (a > 0).')
    assert(p > 0, 'Parameter a is not in the feasible range (0 < p < 1).')
    for i = 1 : length(E)
        out(i) = a*tanh(E(i)^p);
    end
    out = out';
end

% Finite-Time and Predefined-Time Convergence Design for Zeroing Neural Network: Theorem, Method, and Verification
% Section 4-D
function out = Bi_Power(E, a, p, q)
    assert(a > 0, 'Parameter a is not in the feasible range (a > 0).')
    assert(p < 1, 'Parameter a is not in the feasible range (p < 1).')
    assert(q > 1, 'Parameter a is not in the feasible range (q > 1).')
    for i = 1 : length(E)
        out(i) = a*(E(i)^p + E(i)^q);
    end
    out = out';
end
