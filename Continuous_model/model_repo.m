function output = model_repo
    output.OZNN    = @Model_OZNN;
    output.OGNN    = @Model_OGNN;
end

function output = AF_Select(AF_Name, err, t, hyper_params)
    AF = AFs;
    if strcmp(AF_Name, 'linear')
        output = AF.linear(err);
    elseif strcmp(AF_Name, 'powerQ')
        a = hyper_params(1);
        q = hyper_params(2);
        output = AF.powerQ(err, a, q);
    elseif strcmp(AF_Name, 'power')
        output = AF.power(err);
    elseif strcmp(AF_Name, 'PTC')
        Conv_time = hyper_params(1);
        output = AF.PTC(err, t, Conv_time);
    elseif strcmp(AF_Name, 'hs')
        zeta = hyper_params(1);
        output = AF.hs(err, zeta);
    elseif strcmp(AF_Name, 'ps')
        zeta = hyper_params(1);
        m = hyper_params(2);
        output = AF.ps(err, zeta, m);
    elseif strcmp(AF_Name, 'bs')
        zeta = hyper_params(1);
        output = AF.bs(err, zeta);
    elseif strcmp(AF_Name, 'sbp')
        r = hyper_params(1);
        output = AF.sbp(err, r);
    elseif strcmp(AF_Name, 'wsbp')
        r = hyper_params(1);
        k1 = hyper_params(2);
        k2 = hyper_params(3);
        k3 = hyper_params(4);
        output = AF.wsbp(err, r, k1, k2, k3);
    elseif strcmp(AF_Name, 'VAF')
        eta = hyper_params(1);
        w = hyper_params(2);
        a1 = hyper_params(3);
        a2 = hyper_params(4);
        a3 = hyper_params(5);
        a4 = hyper_params(6);
        output = AF.VAF(err, eta, w, a1, a2, a3, a4);
    elseif strcmp(AF_Name, 'TAF')
        eta = hyper_params(1);
        k1 = hyper_params(2);
        k2 = hyper_params(3);
        k3 = hyper_params(4);
        output = AF.TAF(err, eta, k1, k2, k3);
    elseif strcmp(AF_Name, 'bound')
        b = hyper_params(1);
        output = AF.bound(err, b);
    elseif strcmp(AF_Name, 'ball')
        b = hyper_params(1);
        output = AF.ball(err, b);
    elseif strcmp(AF_Name, 'tp')
        a = hyper_params(1);
        p = hyper_params(2);
        output = AF.tp(err, a, p);
    elseif strcmp(AF_Name, 'BiP')
        a = hyper_params(1);
        p = hyper_params(2);
        q = hyper_params(3);
        output = AF.BiP(err, a, p, q);
    end
end

%% System noises define
function output = Noises(kind, Ele_length, t, strength)
    assert(ismember(kind, [0, 1, 2, 3]), 'Kind is not support (kind = 0 (noise free), 1 (constant), 2 (linear tv), or 3 (random)).')
    % Noise Free
    if kind == 0
        output = 0;
    % Constant Noise
    elseif kind == 1
        output = strength;
    % Linear Noise
    elseif kind == 2
        output = strength * t;
    % Bounded Random Noise
    elseif kind == 3
        output = strength * rand(Ele_length, 1);
    else
        output = 0;
    end
end

%% ZNN with activation function
function output=Model_OZNN(t, x, AF, hyper_params, gamma, noise_setting)
    Mat_Vec = Matrix_Vec;
    D = Mat_Vec.D(t);
    w = Mat_Vec.w(t);
    dot_D = Mat_Vec.DotD(t);
    dot_w = Mat_Vec.Dotw(t);
    
    err = D * x + w;

    noise_kind     = noise_setting(1);
    noise_strength = noise_setting(2);
    Ele_length     = length(x);

    % Complex-valued activation method is implemented according to 
    % "Nonlinearly Activated Neural Network for Solving Time-Varying Complex Sylvester Equation" (Equ. 7).
    % "Nonconvex_and_Bound_Constraint_Zeroing_Neural_Network_for_Solving_Time-Varying_Complex-Valued_Quadratic_Programming_Problem
    % (Equ. 8)
    if ~isreal(err)
        err_real = AF_Select(AF, real(err), t, hyper_params);
        err_imag = AF_Select(AF, imag(err), t, hyper_params);
        err = err_real + 1i*err_imag;
    else
        err = AF_Select(AF, err, t, hyper_params);
    end
    
    dotX = pinv(D)*(-gamma*err-dot_D*x-dot_w + Noises(noise_kind, Ele_length, t, noise_strength));
    output = dotX;
    t
end

%% GNN model with activation function
function output=Model_OGNN(t, x, AF, hyper_params, gamma, noise_setting)
    Mat_Vec = Matrix_Vec;
    D = Mat_Vec.D(t);
    w = Mat_Vec.w(t);
    
    err = D * x + w;

    noise_kind     = noise_setting(1);
    noise_strength = noise_setting(2);
    Ele_length     = length(x);

    if ~isreal(err)
        err_real = AF_Select(AF, real(err), t, hyper_params);
        err_imag = AF_Select(AF, imag(err), t, hyper_params);
        err = err_real + 1i*err_imag;
    else
        err = AF_Select(AF, err, t, hyper_params);
    end
    
    dotX = - gamma * D'*(err) + Noises(noise_kind, Ele_length, t, noise_strength);
    output = dotX;
    t
end
