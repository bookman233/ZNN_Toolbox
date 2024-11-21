function output = Matrix_Vec
    output.D    = @Matrix_D;
    output.w    = @Vector_w;
    output.DotD = @Diff_D;
    output.Dotw = @Diff_w;
end

%% A static linear equation example (Only for GNN)
% function D = Matrix_D(t)
%     D = [6.51243521061066, 2.74891421134180;
%          2.74891421134180, 1.36445000945816];
% end
% 
% function w = Vector_w(t)
%     w = ones(2,1);
% end

%% A dynamic linear equation exmple
function D = Matrix_D(t)
    D = [sin(t) cos(t); -cos(t) sin(t)];
end

function w = Vector_w(t)
    w = [-3*sin(5*t) -cos(4*t)]';
end

%% A dynamic complex-valued linear equation example 
% function D = Matrix_D(t)
%     D = [2+sin(5*t),  exp(5*i*t), -sin(5*t)*i, -cos(5*t)*i;
%          exp(-5*i*t), 2+sin(5*t), -cos(5*t)*i, sin(5*t)*i;
%          sin(5*t)*i,  cos(5*t)*i, 0, 0;
%          cos(5*t)*i, -sin(5*t)*i, 0, 0];
% end
% 
% function w = Vector_w(t)
%     w = [exp(-5*i*t), exp(-10*i*t), exp(-10*i*t), exp(-20*i*t)]';
% end


%% Compute the time derivative
function output = Diff_D(t)
    syms u;
    D = Matrix_D(u);
    Dot_D = diff(D);
    u=t;
    output = eval(Dot_D);
end

function output = Diff_w(t)
    syms u;
    w = Vector_w(u);
    Dot_w = diff(w);
    u=t;
    output = eval(Dot_w);
end