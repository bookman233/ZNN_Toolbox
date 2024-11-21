function output = model_repo_with_inte
    output.NTZNN   = @Model_NTZNN;
    output.NTZNNAF = @Model_AFNTZNN;
    output.NTGNN   = @Model_NTGNN;
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

%% ZNN with with integration and activation functions
function output=Model_AFNTZNN(t, x, AF, AF_params, hyperparams, noise_setting)
    Mat_Vec = Matrix_Vec;
    D = Mat_Vec.D(t);
    w = Mat_Vec.w(t);
    dot_D = Mat_Vec.DotD(t);
    dot_w = Mat_Vec.Dotw(t);

    x_length = length(x);
    half = (x_length/2);
    x_cur = x(1:half);
    
    inte  = x(half+1:x_length);
    err   = D * x_cur + w;
    gamma = hyperparams(1);
    mu    = hyperparams(2);

    noise_kind     = noise_setting(1);
    noise_strength = noise_setting(2);
    Ele_length     = length(x_cur);

    AF_One = AF(1);
    AF_Two = AF(2);
    AF_Params_One = AF_params(1,:);
    AF_Params_Two = AF_params(2,:);
    if ~isreal(err)
        err_real = AF_Select(AF_One, real(err), t, AF_Params_One);
        err_imag = AF_Select(AF_One, imag(err), t, AF_Params_One);
        inte_real = AF_Select(AF_Two, real(err + gamma*inte), t, AF_Params_Two);
        inte_imag = AF_Select(AF_Two, imag(err + gamma*inte), t, AF_Params_Two);
        err_AF = err_real + 1i*err_imag;
        inte_AF = inte_real + 1i*inte_imag;
    else
        err = AF_Select(AF_One, err, t, AF_Params_One);
        err_AF = err;
        inte_AF = AF_Select(AF_Two, (err + gamma*inte), t, AF_Params_Two);
    end

    dotX = pinv(D)*(-gamma * err_AF - dot_D*x_cur-dot_w - mu * inte_AF + Noises(noise_kind, Ele_length, t, noise_strength));
    output = [dotX; err];
    norm(err)
    t
end

%% ZNN with with integration
function output=Model_NTZNN(t, x, AF, AF_params, hyperparams, noise_setting)
    Mat_Vec = Matrix_Vec;
    D = Mat_Vec.D(t);
    w = Mat_Vec.w(t);
    dot_D = Mat_Vec.DotD(t);
    dot_w = Mat_Vec.Dotw(t);

    x_length = length(x);
    half = (x_length/2);
    x_cur = x(1:half);
    
    inte = x(half+1:x_length);
    err = D * x_cur + w;
    gamma = hyperparams(1);
    mu    = hyperparams(2);

    noise_kind     = noise_setting(1);
    noise_strength = noise_setting(2);
    Ele_length     = length(x_cur);
    
    dotX = pinv(D)*(-gamma*err-dot_D*x_cur-dot_w - mu*inte + Noises(noise_kind, Ele_length, t, noise_strength));
    output = [dotX;err];
    t
end

%% GNN with with integration
function output=Model_NTGNN(t, x, AF, AF_params, hyperparams, noise_setting)
    Mat_Vec = Matrix_Vec;
    D = Mat_Vec.D(t);
    w = Mat_Vec.w(t);

    x_length = length(x);
    half = (x_length/2);
    x_cur = x(1:half);
    inte = x(half+1:x_length);
    
    err = D * x_cur + w;
    gamma = hyperparams(1);
    mu    = hyperparams(2);

    noise_kind     = noise_setting(1);
    noise_strength = noise_setting(2);
    Ele_length     = length(x_cur);
    
    dotX = - D'*(gamma * err + mu * inte) + Noises(noise_kind, Ele_length, t, noise_strength);
    output = [dotX;err];
    t
end