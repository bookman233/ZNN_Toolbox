function output = ODE_Solvers
    output.RK4       = @ODE_RK4;
    output.RK4_Inte  = @ODE_RK4_Inte;
    output.Euler_Imp = @ODE_ImprovedEuler;
    output.Euler_2O  = @ODE_2OrderEuler; 
end

%% 4-Order Runge-Kutta method for ZNN without integration
function [T,X] = ODE_RK4(Hfun, t, h, x0, AF, hyper_params, gamma, noise_info)
    n = length(t);
    if n == 1
        T = 0:h:t;
    elseif n == 2
        T = t(1):h:t(2);
    else
        T = t;
    end
    T = T';    % 时间变为列向量

    % 计算
    N = length(T);
    x0 = x0.';        % 初值变为列向量  
    m = length(x0);   % 变量个数
    X = zeros(N,m);   % 初始化变量
    dX = zeros(N,m);  % 变量导数
    X(1,:) = x0;
    if ~isreal(x0)
        for k = 2:N
            h = T(k) - T(k-1);
            K1 = Hfun( T(k-1)    , X(k-1,:).', AF, hyper_params, gamma, noise_info);    
            K2 = Hfun( T(k-1)+h/2, X(k-1,:).'+h*K1/2, AF, hyper_params, gamma, noise_info); 
            K3 = Hfun( T(k-1)+h/2, X(k-1,:).'+h*K2/2, AF, hyper_params, gamma, noise_info); 
            K4 = Hfun( T(k-1)+h  , X(k-1,:).'+h*K3, AF, hyper_params, gamma, noise_info); 
            X(k,:) = X(k-1,:).' + (h/6) * ( K1 + 2*K2 + 2*K3 + K4);      
            dX(k-1,:) = (1/6) * (K1 + 2*K2 + 2*K3 + K4);
        end
    else
        for k = 2:N
            h = T(k) - T(k-1);
            K1 = Hfun( T(k-1)    , X(k-1,:)', AF, hyper_params, gamma, noise_info);    
            K2 = Hfun( T(k-1)+h/2, X(k-1,:)'+h*K1/2, AF, hyper_params, gamma, noise_info); 
            K3 = Hfun( T(k-1)+h/2, X(k-1,:)'+h*K2/2, AF, hyper_params, gamma, noise_info); 
            K4 = Hfun( T(k-1)+h  , X(k-1,:)'+h*K3, AF, hyper_params, gamma, noise_info); 
            X(k,:) = X(k-1,:)' + (h/6) * ( K1 + 2*K2 + 2*K3 + K4);      
            dX(k-1,:) = (1/6) * (K1 + 2*K2 + 2*K3 + K4);
        end
    end
end

%% 4-Order Runge-Kutta method for ZNN with integration
function [T,X] = ODE_RK4_Inte(Hfun, t, h, x0, AF, AF_params, hyperparams, noise_info)
    n = length(t);
    if n == 1
        T = 0:h:t;
    elseif n == 2
        T = t(1):h:t(2);
    else
        T = t;
    end
    T = T';    % 时间变为列向量

    % 计算
    N = length(T);
    x0 = x0.';                  % 初值变为列向量  
    m = length(x0);            % 变量个数
    X = zeros(N,m);            % 初始化变量
    dX = zeros(N,m);           % 变量导数
    X(1,:) = x0;
    if ~isreal(x0)
        for k = 2:N
            h = T(k) - T(k-1);
            K1 = Hfun( T(k-1)    , X(k-1,:).', AF, AF_params, hyperparams, noise_info);    
            K2 = Hfun( T(k-1)+h/2, X(k-1,:).'+h*K1/2, AF, AF_params, hyperparams, noise_info); 
            K3 = Hfun( T(k-1)+h/2, X(k-1,:).'+h*K2/2, AF, AF_params, hyperparams, noise_info); 
            K4 = Hfun( T(k-1)+h  , X(k-1,:).'+h*K3, AF, AF_params, hyperparams, noise_info); 
            X(k,:) = X(k-1,:).' + (h/6) * ( K1 + 2*K2 + 2*K3 + K4);      
            dX(k-1,:) = (1/6) * (K1 + 2*K2 + 2*K3 + K4);
        end
    else
        for k = 2:N
            h = T(k) - T(k-1);
            K1 = Hfun( T(k-1)    , X(k-1,:)', AF, AF_params, hyperparams, noise_info);    
            K2 = Hfun( T(k-1)+h/2, X(k-1,:)'+h*K1/2, AF, AF_params, hyperparams, noise_info); 
            K3 = Hfun( T(k-1)+h/2, X(k-1,:)'+h*K2/2, AF, AF_params, hyperparams, noise_info); 
            K4 = Hfun( T(k-1)+h  , X(k-1,:)'+h*K3, AF, AF_params, hyperparams, noise_info); 
            X(k,:) = X(k-1,:)' + (h/6) * ( K1 + 2*K2 + 2*K3 + K4);      
            dX(k-1,:) = (1/6) * (K1 + 2*K2 + 2*K3 + K4);
        end
    end
end

function [T,X,dX] = ODE_ImprovedEuler(Hfun, t, h, x0, gamma, mu)
    if nargin < 4
        error('初始值必须给出');
    end  
    
    % 确定时间节点
    n = length(t);
    if n == 1
        T = 0:h:t;
    elseif n == 2
        T = t(1):h:t(2);
    else
        T = t;
    end
    T = T(:);    % 时间变为列向量
    
    % 计算
    N = length(T);
    x0 = x0(:);  x0 = x0';     % 初值变为行向量  
    m = length(x0);            % 状态量维数
    X = zeros(N,m);            % 初始化状态量
    dX = zeros(N,m);           % 状态导数
    X(1,:) = x0;
    for k = 2:N
        dX(k-1,:) = Hfun( T(k-1), X(k-1,:)', gamma, mu);   
        h = T(k) - T(k-1);
        Xp = X(k-1,:) + h*dX(k-1,:);
        dXp = Hfun( T(k), Xp', gamma, mu);
        X(k,:) = X(k-1,:) + (h/2)*(dX(k-1,:)+dXp');
    end
    dX(N,:) = Hfun( T(N),X(N,:)', gamma, mu);
    
    if nargout == 0
        plot(T,X)
    end
end

function [T,X,dX] = ODE_2OrderEuler(Hfun, t, h, x0, gamma, mu)
    if nargin < 4
        error('初始值必须给出');
    end  
    
    % 确定时间节点
    n = length(t);
    if n == 1
        T = 0:h:t;
    elseif n == 2
        T = t(1):h:t(2);
    else
        T = t;
    end
    T = T(:);    % 时间变为列向量
    
    % 计算
    N = length(T);
    x0 = x0(:);  x0 = x0';     % 初值变为行向量  
    m = length(x0);            % 状态量维数
    X = zeros(N,m);            % 初始化状态量
    dX = zeros(N,m);           % 状态导数
    X(1,:) = x0;
    for k = 2:N
        dX(k-1,:) = Hfun(T(k-1), X(k-1,:)', gamma, mu);   
        h = T(k) - T(k-1);
        if k == 2
            X(k,:) = X(k-1,:) + h*dX(k-1,:);    
        else
            X(k,:) = X(k-2,:) + 2*h*dX(k-1,:);   
        end
    end
    dX(N,:) = Hfun(T(N), X(N,:)', gamma, mu);
    
    if nargout == 0
        plot(T,X)
    end
end
