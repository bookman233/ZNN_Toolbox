# ZNN_Toolbox
Code implementation for various zeroing neural networks and gradient neural networks.
By default, the unconstrained quadratic minimization problem is adopted as an example.

## Environments
- **Matlab 2021b**
- **Matlab Optimization Toolbox**

## Run 
### Continuous model
- Adopt Matlab ode45 to run continuous models without integration: main_model.m.  
`[t, x] = ode45(@model.OZNN, tspan, x0, options, AF, hyper_params, gamma, noise_info);`
- Adopt 4-order Runge-Kutta to run continuous models: main_model.m.  
`t, x] = ODE.RK4(@model.OZNN, tspan, iter_gap, x0, AF, hyper_params, gamma, noise_info);`
