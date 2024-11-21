function output = TV_Params
    output.vp1 = @VPF1;
    output.vp2 = @VPF2;
    output.vp3 = @VPF3;
    output.vp4 = @VPF4;
    output.vp5 = @VPF5;
    output.nac = @NAC;
end

% Varying-Parameter RNN Activated by Finite-Time Functions for Solving Joint-Drift Problems of Redundant Robot Manipulators
function out = VPF1(t, p)
    assert(p > 0, 'Parameter p is not in the feasible range (p > 0).')
    out = t^p + p;
end

% Varying-Parameter RNN Activated by Finite-Time Functions for Solving Joint-Drift Problems of Redundant Robot Manipulators
function out = VPF2(t, p)
    assert(p > 0, 'Parameter p is not in the feasible range (p > 0).')
    out = p^t + p;
end

% A New Varying-Parameter Convergent-Differential Neural-Network for Solving Time-Varying Convex QP Problem Constrained by Linear-Equality
function out = VPF3(t, p)
    assert(p > 0, 'Parameter p is not in the feasible range (p > 0).')
    out = p*exp(t);
end

% A Parameter-Changing and Complex-Valued Zeroing Neural-Network for Finding Solution of Time-Varying Complex Linear Matrix Equations in Finite Time
function out = VPF4(t, p)
    assert(p > 0, 'Parameter p is not in the feasible range (p > 0).')
    if p>0 && p<=1
        out = t^p + p;
    else
        out = p^t + 2*p*t + p;
    end
end

% A Parameter-Changing and Complex-Valued Zeroing Neural-Network for Finding Solution of Time-Varying Complex Linear Matrix Equations in Finite Time
function out = VPF5(t, p)
    assert(p > 0, 'Parameter p is not in the feasible range (p > 0).')
    if p>0 && p<=1
        out = p*exp(t);
    else
        out = p^t + 2*p*t + p;
    end
end

% Norm-Based Adaptive Coefficient ZNN for Solving the Time-Dependent Algebraic Riccati Equation
function out = NAC(E, eta, zeta)
    assert((eta > 0 && zeta>1), 'Parameter eta or zeta is not in the feasible range (eta > 0 and zeta>1).')
    out = (norm(E)^eta)+zeta;
end

