using LeastSquaresOptim
using PyPlot
using DifferentialEquations
using Optim, Flux
using Random,Printf
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
using ForwardDiff, SparseArrays
using Optimization, ProgressMeter
using Zygote, FluxOptTools, Statistics
#Main problem and function in 1D
#In this file, we give the main problem we consider and give the construction of neural neywork with only one hidden layer
#We consider the unconstrained problem
#min_x f(x), where f is a twice-continuously differentiable objective function which maps from Rn into R, and is bounded below
#We use the trust-region methods to solve it, to this end, we need to solve the problem 
#min_{||s||<=Δ_k} m_k(xk+s) = min_{||s||<=Δ_k} f(xk) + J(xk)'*s + 1/2 s'*H_k*s, 
#where J(xk) is the gradient of f,i.e. ∇f(xk); H_k is the Hessian matrix or its approximation, for instance, J(xk)'*J(xk)
#Indeed, at each iteration, a step s is computed as an approximate solution of
#min_{||s||<=Δ_k} f(xk)+J(xk)'*s+1/2 s'*H_k*s+1/2 λ||s||^2, where λ>0 is a regularization parameter.
#In particular, we use the above optimization problem to train the neural network, which can approximate the solution of PDEs.
#Consider the PDEs as follows:
#D(x,u(x))=g_1(x), x\in Ω, BC(x,u(x))=g_2(x), x\in ∂Ω,
#where Ω is an open connected subset of Rn, n>=1, ∂Ω is the boundary of Ω, D is a differential operator, BC is an operator defining the boundary conditions.
#g_1,g_2 :Rn -> R are given functions.
#First, let's just consider an one-hidden-layer neural network with r nodes in the hidden layer. 
#Let v:=[v_1,...,v_r]^T be the output weights, w_i:=[w_i1,...,w_ir]^T be the input weights, for any i=1,...,n,
#b:=[b_1,b_2,...,b_r]^T be the biases of the hidden nodes, d be the output bias, p:=[v,w_1,...,w_n,b,d]^T be the stacked vectors of weights and biases.
#We denote the output as N(p,x;r), which can be written as
#N(p,x;r)=∑_{i=1}^{r}v_i * σ(∑_{j=1}^{n}w_{ji}x_j+b_i)+d
#Training phase then can be expressed as
#min_{p} L(p,x) = 1/(2*N_D)\sum_{i=1}^{N_D}||D(x_Di,N(p,x_Di;r))-g_1(x_Di)||^2+(λ_p)/(2*N_BC)\sum_{i=1}^{N_BC}||BC(x_BCi,N(p,x_BCi;r))-g_2(x_BCi)||^2
#where {x_BCi}_{i=1}^{N_BC} denote the boundary training data on u(x), and {x_Di}_{i=1}^{N_D} specify the collocations points for D(x,u(x)), λ_p>0 is the penalty parameter

###Arguments

# 'obj_func::AbstractNLPModel': the objective function we want to optimize
# 'λk': a regularization parameter 
# 'options::SolverOptions': a structure containing algorithmic Parameters

###keyword Arguments
#'p0::AbstractVector': an initial guess
#'tol': relative stopping tolerence, i.e. ||∇_s m_k(xk,sk)+λk*sk||/||sk|| <= tol
###Return Arguments
#'sk': the final iterate

###Reference 
# H. Calandra, S. Gratton, E. Riccietti and X. Vasseur. On a multilevel Levenberg Marquardt method for the training of artificial neural networks and its application to the solution of partial differential equations
using LeastSquaresOptim
using PyPlot
using DifferentialEquations
using Optim, Flux
using Random,Printf
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
using ForwardDiff, SparseArrays
using Optimization, ProgressMeter
using Zygote, FluxOptTools, Statistics

using ForwardDiff
using LinearAlgebra

function possion_1d(real_solution::Function)
    #@parameters x
    #@variables u(..)
    u(x) = real_solution(x)
    u_grad(x) = ForwardDiff.derivative(x -> u(x),x)
    g_1(x) = -ForwardDiff.derivative(x -> u_grad(x),x)
    g_2(x) = real_solution(x)
    return u,g_1,g_2
end

function cos_v(ν)
    function cos_vx(x)
        return cos(ν*x)
    end
    return cos_vx
end

function constant(a)
    function a_func(x)
        return (-a/2)*x^2
    end
    return a_func
end

function one_hidden_layer_nn(
    input_weights,#input_weights is nodes_num*1
    input_biases,#input_biases is nodes_num*1
    output_weights,#output_weights is nodes_num*1
    output_bias,#output_weights is a scalar
    data,#sampled nD-PDEs variables matrix, 
    σ,#activation function,
    )
    inner =   input_weights * transpose(data)  .+ input_biases
    active = σ.(inner)
    output = transpose(output_weights) * active .+ output_bias
    return output'
end
nn_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights ->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
nn_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
nn_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
nn_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_bias -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)

function second_derivative(σ)
    σ_1(x) = ForwardDiff.derivative(σ,x)
    σ_2(x) = ForwardDiff.derivative(σ_1,x)
    return σ_2
end

function nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ)
    inner =   input_weights * transpose(data)  .+ input_biases
    active = second_derivative(σ).(inner)
    outweight = output_weights.*input_weights.^2
    output = transpose(outweight) * active
    return output'
end 

nn_lap1d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights ->nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
nn_lap1d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases -> nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
nn_lap1d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights -> nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
nn_lap1d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_bias -> nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)



function obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
    g_1 = possion_1d(constant(1))[2] #the right-hand side could choose other PDEs
    g_2 = possion_1d(constant(1))[3]
    N_u = size(data)[1]-2
    g_1_data = g_1.(data[2:end-1])
    loss = norm(g_1_data+nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ),2)^2
    loss = loss/(2*N_u)
    λ_p = 0.1
    penalty = norm(g_2(data[1])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[1],2)^2+norm(g_2(data[end])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[end],2)^2
    penalty = λ_p*penalty/4
    return loss+penalty
end

obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_weights->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_biases->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_weights->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_bias->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)
grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) = vcat(obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ)) 

function Bk(input_weights,input_biases,output_weights,output_bias,data,σ)
    F_w = zeros(1,size(input_weights)[1])
    F_b = zeros(1,size(input_biases)[1])
    F_v = zeros(1,size(output_weights)[1])
    F_d = zeros(1,size(output_bias)[1])
    λ_p = 0.1*size(data)[1]
    for i in 2:size(data)[1]-1
    F_w_i = ForwardDiff.jacobian(input_weights ->nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data[i],σ),input_weights)
    F_b_i = ForwardDiff.jacobian(input_biases -> nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data[i],σ),input_biases)
    F_v_i = ForwardDiff.jacobian(output_weights -> nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data[i],σ),output_weights)
    F_d_i = ForwardDiff.jacobian(output_bias -> nn_lap1D_x(input_biases,input_biases,output_weights,output_bias,data[i],σ),output_bias)
    F_w += F_w_i
    F_b += F_b_i
    F_v += F_v_i
    F_d += F_d_i
    end
    loss_term = vcat(F_w',F_b',F_v',F_d')*hcat(F_w,F_b,F_v,F_d)
    N_w_0 = ForwardDiff.jacobian(input_weights ->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[1],σ),input_weights)
    N_b_0 = ForwardDiff.jacobian(input_biases -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[1],σ),input_biases)
    N_v_0 = ForwardDiff.jacobian(output_weights -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[1],σ),output_weights)
    N_d_0 = ForwardDiff.jacobian(output_bias -> one_hidden_layer_nn(input_biases,input_biases,output_weights,output_bias,data[1],σ),output_bias)
    N_w_end = ForwardDiff.jacobian(input_weights ->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[end],σ),input_weights)
    N_b_end = ForwardDiff.jacobian(input_biases -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[end],σ),input_biases)
    N_v_end = ForwardDiff.jacobian(output_weights -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[end],σ),output_weights) 
    N_d_end = ForwardDiff.jacobian(output_bias -> one_hidden_layer_nn(input_biases,input_biases,output_weights,output_bias,data[end],σ),output_bias)
    dN_0 = hcat(N_w_0,N_b_0,N_v_0,N_d_0)
    dN_end = hcat(N_w_end,N_b_end,N_v_end,N_d_end)
    penalty_term = dN_0'*dN_0+dN_end'*dN_end
    return loss_term/(size(data)[1]-2)+penalty_term*λ_p/2
end






function mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1*length(data)
    first_term = 0
    for i in 2:length(data)-1
        F1 = F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1 = J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        first_term += norm(F1,2)^2+(2*(J1'*F1)'*s)[1]+s'*J1'*J1*s
    end
    F20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    F2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    J2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = norm(F20,2)^2+(2*(J20'*F20)'*s)[1]+s'*J20'*J20*s+norm(F2e,2)^2+(2*(J2e'*F2e)'*s)[1]+s'*J2e'*J2e*s
    return first_term/(2*N_D)+λ_p*second_term/(2*N_BC)+λ*norm(s,2)^2
end

function grad_f(input_weights,input_biases,output_weights,output_bias,data,σ)
    first = zeros(3*size(input_weights)[1]+1)
    for i in 2:length(data)-1
        first .+= J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)'*F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
    end
    return first/(length(data)-2)+0.1*size(data)[1]*(J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)'*F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)+J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)'*F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ))/2
end














function F_1(input_weights,input_biases,output_weights,output_bias,data,σ)
    g_1 = possion_1d(constant(1))[2]
    if length(data) == 1
        return g_1(data).+nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ)
    else
        return g_1.(data)+nn_lap1D_x(input_weights,input_biases,output_weights,output_bias,data,σ)
    end
end

function F_2(input_weights,input_biases,output_weights,output_bias,data,σ)
    g_2 = possion_1d(constant(1))[3]
    if length(data) == 1
        return g_2(data).-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)
    else
        return g_2.(data)-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)
    end
end



function J_1(input_weights,input_biases,output_weights,output_bias,data,σ)
    F_1w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_1b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_1v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    F_1d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_bias->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)
    return hcat(F_1w(input_weights,input_biases,output_weights,output_bias,data,σ),F_1b(input_weights,input_biases,output_weights,output_bias,data,σ),F_1v(input_weights,input_biases,output_weights,output_bias,data,σ),F_1d(input_weights,input_biases,output_weights,output_bias,data,σ))
end

function J_2(input_weights,input_biases,output_weights,output_bias,data,σ)
    F_2w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_2b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_2v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    F_2d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_bias->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)
    return hcat(F_2w(input_weights,input_biases,output_weights,output_bias,data,σ),F_2b(input_weights,input_biases,output_weights,output_bias,data,σ),F_2v(input_weights,input_biases,output_weights,output_bias,data,σ),F_2d(input_weights,input_biases,output_weights,output_bias,data,σ))
end
function mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1
    first_term = 0
    for i in 2:length(data)-1
        F1 = F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1 = J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        first_term += norm(F1,2)^2+(2*(J1'*F1)'*s)[1]+s'*J1'*J1*s
    end
    F20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    F2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    J2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = norm(F20,2)^2+(2*(J20'*F20)'*s)[1]+s'*J20'*J20*s+norm(F2e,2)^2+(2*(J2e'*F2e)'*s)[1]+s'*J2e'*J2e*s
    return first_term/(2*N_D)+λ_p*second_term/(2*N_BC)+λ*norm(s,2)^2
end

function line_mk(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1
    first_term = zeros(size(s)[1])
    for i in 2:length(data)-1
        F1 = F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1 = J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        first_term .+= J1'*F1+J1'*J1*s
    end
    F20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    F2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    J2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = J20'*F20+J20'*J20*s+J2e'*F2e+J2e'*J2e*s
    return first_term/(N_D)+λ_p*second_term/(N_BC)+λ*s
end

function line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1
    first_term = zeros(s_size,s_size)
    for i in 2:length(data)-1
        #F1 = F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1 = J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        first_term .+= J1'*J1
    end
    #F20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    #F2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    J2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = J20'*J20+J2e'*J2e
    return first_term/(N_D)+λ_p*second_term/N_BC+λ*Matrix{Float64}(I, s_size, s_size) 
end
function line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1
    first_term = zeros(s_size)
    for i in 2:length(data)-1
        F1 = F_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1 = J_1(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        first_term .+= J1'*F1
    end
    F20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    F2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    J2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = J20'*F20+J2e'*F2e
    return first_term/(N_D)+λ_p*second_term/(N_BC)
end


function LM_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-4
    s_size = 3*size(output_weights)[1]+1
    para_size = size(output_weights)[1]
    #give an initial step sk
    s = 0.001*ones(s_size)
    iteration = 0
    max_iteration = 1000
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) >= ϵ
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
        #change to solve the linear system
        #s =  Optim.minimizer(optimize(mk, s, LBFGS(); autodiff =:forward))
        #@show s
        
        s = vec(cholesky(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ))\((-1).*(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ))))
       
        #@show s
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
        @show fk(s) fk(zeros(s_size))
        ρkn = fk(zeros(s_size))-fk(s)
        ρkd = mk(zeros(s_size))-mk(s)
        ρ = ρkn/ρkd
        @show ρkn ρkd ρ
        if ρ >= η1
            input_weights = input_weights.+s[1:para_size]
            input_biases = input_biases.+s[para_size+1:2*para_size]
            output_weights = output_weights.+s[2*para_size+1:3*para_size]
            output_bias = output_bias.+s[end]
            if ρ >= η2
                λ = max(λ_min,γ2*λ)
            else
                λ = max(λ_min,γ1*λ)
            end
        else
            input_weights = input_weights
            input_biases = input_biases
            output_weights = output_weights
            output_bias = output_bias
            λ = γ3*λ
        end
        iteration += 1
    end
    println("Iteration: $iteration")
    @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
    return input_weights,input_biases,output_weights,output_bias
end

function LM_1D_CG(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-4
    s_size = 3*size(output_weights)[1]+1
    para_size = size(output_weights)[1]
    #give an initial step sk
    s = 0.001*ones(s_size)
    iteration = 0
    max_iteration = 1000
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) >= ϵ
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
        #change to solve the linear system
        #s =  Optim.minimizer(optimize(mk, s, LBFGS(); autodiff =:forward))
        #@show s
        
        s = cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
        iteration += cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2].niter
        #@show s
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
        @show fk(s) fk(zeros(s_size))
        ρkn = fk(zeros(s_size))-fk(s)
        ρkd = mk(zeros(s_size))-mk(s)
        ρ = ρkn/ρkd
        @show ρkn ρkd ρ
        if ρ >= η1
            input_weights = input_weights.+s[1:para_size]
            input_biases = input_biases.+s[para_size+1:2*para_size]
            output_weights = output_weights.+s[2*para_size+1:3*para_size]
            output_bias = output_bias.+s[end]
            if ρ >= η2
                λ = max(λ_min,γ2*λ)
            else
                λ = max(λ_min,γ1*λ)
            end
        else
            input_weights = input_weights
            input_biases = input_biases
            output_weights = output_weights
            output_bias = output_bias
            λ = γ3*λ
        end
    end
    println("Iteration: $iteration")
    @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
    return input_weights,input_biases,output_weights,output_bias
end

function LM_1D_CGLS(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-4
    s_size = 3*size(output_weights)[1]+1
    para_size = size(output_weights)[1]
    #give an initial step sk
    s = 0.001*ones(s_size)
    iteration = 0
    max_iteration = 1000
    ρkn = 0.5
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) >= ϵ && ρkn != 0 
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
        
        s = cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
       iteration +=cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2].niter
     
        #@show s
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
        @show fk(s) fk(zeros(s_size))
        ρkn = fk(zeros(s_size))-fk(s)
        ρkd = mk(zeros(s_size))-mk(s)
        ρ = ρkn/ρkd
        @show ρkn ρkd ρ
        if ρ >= η1
            input_weights = input_weights.+s[1:para_size]
            input_biases = input_biases.+s[para_size+1:2*para_size]
            output_weights = output_weights.+s[2*para_size+1:3*para_size]
            output_bias = output_bias.+s[end]
            if ρ >= η2
                λ = max(λ_min,γ2*λ)
            else
                λ = max(λ_min,γ1*λ)
            end
        else
            input_weights = input_weights
            input_biases = input_biases
            output_weights = output_weights
            output_bias = output_bias
            λ = γ3*λ
        end
    end
    println("Iteration: $iteration")
    @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
    return input_weights,input_biases,output_weights,output_bias
end
#Test##
x = collect(LinRange(0,1,41))
yreal = -x.^2/2


#r = 100 with σ=sigmoid, x=LinRange(0,1,41)
using CSV, DataFrames
rw_lu_100 = reshape(CSV.read("rw_lu_100.csv",DataFrame)[!,1],(100,1))
rb_lu_100 = CSV.read("rb_lu_100.csv",DataFrame)[!,1]
rv_lu_100 = reshape(CSV.read("rv_lu_100.csv",DataFrame)[!,1],(100,1))
rd_lu_100 = CSV.read("rd_lu_100.csv",DataFrame)[!,1]
#CG#
@time begin
    lrw_lu_100_cg, lrb_lu_100_cg, lrv_lu_100_cg, lrd_lu_100_cg = LM_1D_CG(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
end
yhat_cg100 = one_hidden_layer_nn(lrw_lu_100_cg,lrb_lu_100_cg,lrv_lu_100_cg,lrd_lu_100_cg,x,sigmoid)
error_100_const_cg = norm(vec(yhat_cg100.-yreal),2)/41
#write into the data file and plot
cw1cg = DataFrame(lrw_lu_100_cg,:auto)
CSV.write("lrw_lu_100_cg.csv",cw1cg)
cb1cg = DataFrame([lrb_lu_100_cg],:auto)
CSV.write("lrb_lu_100_cg.csv",cb1cg)
cv1cg = DataFrame(lrv_lu_100_cg,:auto)
CSV.write("lrv_lu_100_cg.csv",cv1cg)
cd1cg = DataFrame([lrd_lu_100_cg],:auto)
CSV.write("lrd_lu_100_cg.csv",cd1cg)
cy1cg = DataFrame(yhat_cg100,:auto)
CSV.write("lyhat_cg_100.csv",cy1cg)

using Plots
Plots.plot(x,yhat,label="NN(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#cgls
@time begin
    lrw_lu_100_cg, lrb_lu_100_cg, lrv_lu_100_cg, lrd_lu_100_cg = LM_1D_CGLS(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
end
yhat_cgls100 = one_hidden_layer_nn(lrw_lu_100_cgls,lrb_lu_100_cgls,lrv_lu_100_cgls,lrd_lu_100_cgls,x,sigmoid)
error_100_const_cgls = norm(vec(yhat_cgls100.-yreal),2)/41
#write into the data file and plot
cw1cgls = DataFrame(lrw_lu_100_cgls,:auto)
CSV.write("lrw_lu_100_cgls.csv",cw1cgls)
cb1cgls = DataFrame([lrb_lu_100_cgls],:auto)
CSV.write("lrb_lu_100_cgls.csv",cb1cgls)
cv1cgls = DataFrame(lrv_lu_100_cgls,:auto)
CSV.write("lrv_lu_100_cgls.csv",cv1cgls)
cd1cgls = DataFrame([lrd_lu_100_cgls],:auto)
CSV.write("lrd_lu_100_cgls.csv",cd1cgls)
cy1cgls = DataFrame(yhat_cgls100,:auto)
CSV.write("lyhatcgls.csv",cy1cgls)

using Plots
Plots.plot(x,yhat_cgls100,label="NN_CGLS(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)


#r = 300 with σ=sigmoid, x=LinRange(0,1,41)
rw_lu_300 = reshape(CSV.read("rw_lu_300.csv",DataFrame)[!,1],(300,1))
rb_lu_300 = CSV.read("rb_lu_300.csv",DataFrame)[!,1]
rv_lu_300 = reshape(CSV.read("rv_lu_300.csv",DataFrame)[!,1],(300,1))
rd_lu_300 = CSV.read("rd_lu_300.csv",DataFrame)[!,1]

#CG#
@time begin
    lrw_lu_300_cg, lrb_lu_300_cg, lrv_lu_300_cg, lrd_lu_300_cg = LM_1D_CG(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,sigmoid)
end
yhat_300cg = one_hidden_layer_nn(lrw_lu_300_cg,lrb_lu_300_cg,lrv_lu_300_cg,lrd_lu_300_cg,x,sigmoid)
error_300_const_cg = norm(vec(yhat_300cg.-yreal),2)/41
#write into the data file and plotting
cw3cg = DataFrame(lrw_lu_300_cg,:auto)
CSV.write("lrw_lu_300_cg.csv",cw3cg)
cb3cg = DataFrame([lrb_lu_300_cg],:auto)
CSV.write("lrb_lu_300_cg.csv",cb3cg)
cv3cg = DataFrame(lrv_lu_300_cg,:auto)
CSV.write("lrv_lu_300_cg.csv",cv3cg)
cd3cg = DataFrame([lrd_lu_300_cg],:auto)
CSV.write("lrd_lu_300_cg.csv",cd3cg)
cy3cg = DataFrame(yhat_300cg,:auto)
CSV.write("lyhat_CG_300.csv",cy3cg)

Plots.plot(x,yhat_300cg,label="NN_CG(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#CGLS#
@time begin
    lrw_lu_300_cgls, lrb_lu_300_cgls, lrv_lu_300_cgls, lrd_lu_300_cgls = LM_1D_CGLS(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,sigmoid)
end

yhat_300cgls = one_hidden_layer_nn(lrw_lu_300_cgls, lrb_lu_300_cgls, lrv_lu_300_cgls, lrd_lu_300_cgls,x,sigmoid)
error_300_const_cgls = norm(vec(yhat_300cgls.-yreal),2)/41
#write into the data file and plotting
cw3cgls = DataFrame(lrw_lu_300_cgls,:auto)
CSV.write("lrw_lu_300_cgls.csv",cw3cgls)
cb3cgls = DataFrame([lrb_lu_300_cgls],:auto)
CSV.write("lrb_lu_300_cgls.csv",cb3cgls)
cv3cgls = DataFrame(lrv_lu_300_cgls,:auto)
CSV.write("lrv_lu_300_cgls.csv",cv3cgls)
cd3cgls = DataFrame([lrd_lu_300_cgls],:auto)
CSV.write("lrd_lu_300_cgls.csv",cd3cgls)
cy3cgls = DataFrame(yhat_300cgls,:auto)
CSV.write("lyhatcgls_300.csv",cy3cgls)

Plots.plot(x,yhat_300cgls,label="NN_CGLS_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#r=500
rw_lu_500 = reshape(CSV.read("rw_lu_500.csv",DataFrame)[!,1],(500,1))
rb_lu_500 = CSV.read("rb_lu_500.csv",DataFrame)[!,1]
rv_lu_500 = reshape(CSV.read("rv_lu_500.csv",DataFrame)[!,1],(500,1))
rd_lu_500 = CSV.read("rd_lu_500.csv",DataFrame)[!,1]

#CG#
@time begin
    lrw_lu_500_cg, lrb_lu_500_cg, lrv_lu_500_cg, lrd_lu_500_cg = LM_1D_CG(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
end
yhat_500cg = one_hidden_layer_nn(lrw_lu_500_cg,lrb_lu_500_cg,lrv_lu_500_cg,lrd_lu_500_cg,x,sigmoid)
error_500_const_cg = norm(vec(yhat_500cg.-yreal),2)/41
#write into a data file and plotting
cw5cg = DataFrame(lrw_lu_500_cg,:auto)
CSV.write("lrw_lu_500_cg.csv",cw5cg)
cb5cg = DataFrame([lrb_lu_500_cg],:auto)
CSV.write("lrb_lu_500_cg.csv",cb5cg)
cv5cg = DataFrame(lrv_lu_500_cg,:auto)
CSV.write("lrv_lu_500_cg.csv",cv5cg)
cd5cg = DataFrame([lrd_lu_500_cg],:auto)
CSV.write("lrd_lu_500_cg.csv",cd5cg)
cy5cg = DataFrame(yhat_500cg,:auto)
CSV.write("lyhat_CG_500.csv",cy5cg)

Plots.plot(x,yhat_500cg,label="NN_CG_500(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#CGLS#
@time begin
    lrw_lu_500_cgls, lrb_lu_500_cgls, lrv_lu_500_cgls, lrd_lu_500_cgls = LM_1D_CGLS(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
end

yhat_500cgls = one_hidden_layer_nn(lrw_lu_500_cgls,lrb_lu_500_cgls,lrv_lu_500_cgls,lrd_lu_500_cgls,x,sigmoid)
error_500_const_cgls = norm(vec(yhat_500cgls.-y),2)/41
#write into a data file and plotting
cw5cgls = DataFrame(lrw_lu_500_cgls,:auto)
CSV.write("lrw_lu_500_cgls.csv",cw5cgls)
cb5cgls = DataFrame([lrb_lu_500_cgls],:auto)
CSV.write("lrb_lu_500_cgls.csv",cb5cgls)
cv5cgls = DataFrame(lrv_lu_500_cgls,:auto)
CSV.write("lrv_lu_500_cgls",cv5cgls)
cd5cgls = DataFrame([lrd_lu_500_cgls],:auto)
CSV.write("lrd_lu_500_cgls.csv",cd5cgls)
ny5cgls = DataFrame(yp_cos500cgls,:auto)
CSV.write("lyhatcgls_500.csv",cy5cgls)

Plots.plot(x,yhat_500cgls,label="NN_CGLS500(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#Test with activation function being identity

#r=100
#CG
@time begin
    lrw_x41_100_cgi, lrb_x41_100_cgi, lrv_x41_100_cgi, lrd_x41_100_cgi = LM_1D_CG(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,identity)
end
lyhat_x41_cg100 = one_hidden_layer_nn(lrw_x41_100_cgi,lrb_x41_100_cgi,lrv_x41_100_cgi,lrd_x41_100_cgi,x,identity)
error_100_const_cgi = norm(vec(lyhat_x41_cg100.-yreal),2)/41
#write into the data file and plot
cw1cgi = DataFrame(lrw_x41_100_cgi,:auto)
CSV.write("lrw_x41_100_cgi.csv",cw1cgi)
cb1cgi = DataFrame([lrb_x41_100_cgi],:auto)
CSV.write("lrb_x41_100_cgi.csv",cb1cgi)
cv1cgi = DataFrame(lrv_x41_100_cgi,:auto)
CSV.write("lrv_x41_100_cgi.csv",cv1cgi)
cd1cgi = DataFrame([lrd_x41_100_cgi],:auto)
CSV.write("lrd_x41_100_cgi.csv",cd1cgi)
cy1cgi = DataFrame(lyhat_x41_cg100,:auto)
CSV.write("lyhat_x41_CG_100.csv",cy1cgi)

using Plots
Plots.plot(x,lyhat_x41_cg100,label="NNi_CG_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x41_100_cglsi, lrb_x41_100_cglsi, lrv_x41_100_cglsi, lrd_x41_100_cglsi = LM_1D_CGLS(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,identity)
end
lyhat_x41_cgls100 = one_hidden_layer_nn(lrw_x41_100_cglsi,lrb_x41_100_cglsi,lrv_x41_100_cglsi,lrd_x41_100_cglsi,x,identity)
error_100_const_cglsi = norm(vec(lyhat_x41_cgls100.-yreal),2)/41
#write into the data file and plot
cw1cglsi = DataFrame(lrw_x41_100_cglsi,:auto)
CSV.write("lrw_x41_100_cglsi.csv",cw1cglsi)
cb1cglsi = DataFrame([lrb_x41_100_cglsi],:auto)
CSV.write("lrb_x41_100_cglsi.csv",cb1cglsi)
cv1cglsi = DataFrame(lrv_x41_100_cglsi,:auto)
CSV.write("lrv_x41_100_cglsi.csv",cv1cglsi)
cd1cglsi = DataFrame([lrd_x41_100_cglsi],:auto)
CSV.write("lrd_x41_100_cglsi.csv",cd1cglsi)
cy1cglsi = DataFrame(lyhat_x41_cgls100,:auto)
CSV.write("lyhat_x41_CGLS_100.csv",cy1cglsi)

using Plots
Plots.plot(x,lyhat_x41_cgls100,label="NNi_CGLS_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#r=300

#CG#
@time begin
    lrw_x41_300_cgi, lrb_x41_300_cgi, lrv_x41_300_cgi, lrd_x41_300_cgi = LM_1D_CG(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,identity)
end
lyhat_x41_cg300 = one_hidden_layer_nn(lrw_x41_300_cgi,lrb_x41_300_cgi,lrv_x41_300_cgi,lrd_x41_300_cgi,x,identity)
error_300_const_cgi = norm(vec(lyhat_x41_cg300.-yreal),2)/41
#write into the data file and plot
cw3cgi = DataFrame(lrw_x41_300_cgi,:auto)
CSV.write("lrw_x41_300_cgi.csv",cw3cgi)
cb3cgi = DataFrame([lrb_x41_300_cgi],:auto)
CSV.write("lrb_x41_300_cgi.csv",cb3cgi)
cv3cgi = DataFrame(lrv_x41_300_cgi,:auto)
CSV.write("lrv_x41_300_cgi.csv",cv3cgi)
cd3cgi = DataFrame([lrd_x41_300_cgi],:auto)
CSV.write("lrd_x41_300_cgi.csv",cd3cgi)
cy3cgi = DataFrame(lyhat_x41_cg300,:auto)
CSV.write("lyhat_x41_CG_300.csv",cy3cgi)

using Plots
Plots.plot(x,lyhat_x41_cg300,label="NNi_CG_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,yreal,label="real_solution",linewidth=2.0)

#CGLS#
@time begin
    lrw_x41_300_cglsi, lrb_x41_300_cglsi, lrv_x41_300_cglsi, lrd_x41_300_cglsi = LM_1D_CGLS(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,identity)
end
lyhat_x41_cgls300 = one_hidden_layer_nn(lrw_x41_300_cglsi,lrb_x41_300_cglsi,lrv_x41_300_cglsi,lrd_x41_300_cglsi,x,identity)
error_300_const_cglsi = norm(vec(lyhat_x41_cgls300.-yreal),2)/41
#write into the data file and plot
cw3cglsi = DataFrame(lrw_x41_300_cglsi,:auto)
CSV.write("lrw_x41_300_cglsi.csv",cw3cglsi)
cb3cglsi = DataFrame([lrb_x41_300_cglsi],:auto)
CSV.write("lrb_x41_300_cglsi.csv",cb3cglsi)
cv3cglsi = DataFrame(lrv_x41_300_cglsi,:auto)
CSV.write("lrv_x41_300_cglsi.csv",cv3cglsi)
cd3cglsi = DataFrame([lrd_x41_300_cglsi],:auto)
CSV.write("lrd_x41_300_cglsi.csv",cd3cglsi)
cy3cglsi = DataFrame(lyhat_x41_cgls300,:auto)
CSV.write("lyhat_x41_CGLS_300.csv",cy3cglsi)

using Plots
Plots.plot(x,lyhat_x41_cgls300,label="NNi_CGLS_300(x)",seriestype=:scatter)
show()




#Test with x=LinRange(0,1,301) and sigmoid being the activation function
x = collect(LinRange(0,1,301))
y = -x.^2/2
#r=100
#CG
@time begin
    lrw_x301_100_cg, lrb_x301_100_cg, lrv_x301_100_cg, lrd_x301_100_cg = LM_1D_CG(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
end
lyhat_x301_cg100 = one_hidden_layer_nn(lrw_x301_100_cg,lrb_x301_100_cg,lrv_x301_100_cg,lrd_x301_100_cg,x,sigmoid)
error_100_const_cg301 = norm(vec(lyhat_x301_cg100.-y),2)/301
#write into the data file and plot
cw1cg301 = DataFrame(lrw_x301_100_cg,:auto)
CSV.write("lrw_x301_100_cg.csv",cw1cg301)
cb1cg301 = DataFrame([lrb_x301_100_cg],:auto)
CSV.write("lrb_x301_100_cg.csv",cb1cg301)
cv1cg301 = DataFrame(lrv_x301_100_cg,:auto)
CSV.write("lrv_x301_100_cg.csv",cv1cg301)
cd1cg301 = DataFrame([lrd_x301_100_cg],:auto)
CSV.write("lrd_x301_100_cg.csv",cd1cg301)
cy1cg301 = DataFrame(lyhat_x301_cg100,:auto)
CSV.write("lyhat_x301_CG_100.csv",cy1cg301)

using Plots
Plots.plot(x,lyhat_x301_cg100,label="NN_CG_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x301_100_cgls, lrb_x301_100_cgls, lrv_x301_100_cgls, lrd_x301_100_cgls = LM_1D_CGLS(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
end
lyhat_x301_cgls100 = one_hidden_layer_nn(lrw_x301_100_cgls,lrb_x301_100_cgls,lrv_x301_100_cgls,lrd_x301_100_cgls,x,sigmoid)
error_100_const_cgls301 = norm(vec(lyhat_x301_cgls100.-y),2)/301
#write into the data file and plot
cw1cgls301 = DataFrame(lrw_x301_100_cgls,:auto)
CSV.write("lrw_x301_100_cgls.csv",cw1cgls301)
cb1cgls301 = DataFrame([lrb_x301_100_cgls],:auto)
CSV.write("lrb_x301_100_cgls.csv",cb1cgls301)
cv1cgls301 = DataFrame(lrv_x301_100_cgls,:auto)
CSV.write("lrv_x301_100_cgls.csv",cv1cgls301)
cd1cgls301 = DataFrame([lrd_x301_100_cgls],:auto)
CSV.write("lrd_x301_100_cgls.csv",cd1cgls301)
cy1cgls301 = DataFrame(lyhat_x301_cgls100,:auto)
CSV.write("lyhat_x301_CGLS_100.csv",cy1cgls301)

using Plots
Plots.plot(x,lyhat_x301_cgls100,label="NN_CGLS_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)


#r=300
#CG
@time begin
    lrw_x301_300_cg, lrb_x301_300_cg, lrv_x301_300_cg, lrd_x301_300_cg = LM_1D_CG(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,sigmoid)
end
lyhat_x301_cg300 = one_hidden_layer_nn(lrw_x301_300_cg,lrb_x301_300_cg,lrv_x301_300_cg,lrd_x301_300_cg,x,sigmoid)
error_300_const_cg301 = norm(vec(lyhat_x301_cg300.-y),2)/301
#write into the data file and plot
cw3cg301 = DataFrame(lrw_x301_300_cg,:auto)
CSV.write("lrw_x301_300_cg.csv",cw3cg301)
cb3cg301 = DataFrame([lrb_x301_300_cg],:auto)
CSV.write("lrb_x301_300_cg.csv",cb3cg301)
cv3cg301 = DataFrame(lrv_x301_300_cg,:auto)
CSV.write("lrv_x301_300_cg.csv",cv3cg301)
cd3cg301 = DataFrame([lrd_x301_300_cg],:auto)
CSV.write("lrd_x301_300_cg.csv",cd3cg301)
cy3cg301 = DataFrame(lyhat_x301_cg300,:auto)
CSV.write("lyhat_x301_CG_300.csv",cy3cg301)

using Plots
Plots.plot(x,lyhat_x301_cg300,label="NN_CG_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x301_300_cgls, lrb_x301_300_cgls, lrv_x301_300_cgls, lrd_x301_300_cgls = LM_1D_CGLS(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,sigmoid)
end
lyhat_x301_cgls300 = one_hidden_layer_nn(lrw_x301_300_cgls,lrb_x301_300_cgls,lrv_x301_300_cgls,lrd_x301_300_cgls,x,sigmoid)
error_300_const_cgls301 = norm(vec(lyhat_x301_cgls300.-y),2)/301
#write into the data file and plot
cw3cgls301 = DataFrame(lrw_x301_300_cgls,:auto)
CSV.write("lrw_x301_300_cgls.csv",cw3cgls301)
cb3cgls301 = DataFrame([lrb_x301_300_cgls],:auto)
CSV.write("lrb_x301_300_cgls.csv",cb3cgls301)
cv3cgls301 = DataFrame(lrv_x301_300_cgls,:auto)
CSV.write("lrv_x301_300_cgls.csv",cv3cgls301)
cd3cgls301 = DataFrame([lrd_x301_300_cgls],:auto)
CSV.write("lrd_x301_300_cgls.csv",cd3cgls301)
cy3cgls301 = DataFrame(lyhat_x301_cgls300,:auto)
CSV.write("lyhat_x301_CGLS_300.csv",cy3cgls301)

using Plots
Plots.plot(x,lyhat_x301_cgls300,label="NN_CGLS_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#r=500

#CG
@time begin
    lrw_x301_500_cg, lrb_x301_500_cg, lrv_x301_500_cg, lrd_x301_500_cg = LM_1D_CG(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
end
lyhat_x301_cg500 = one_hidden_layer_nn(lrw_x301_500_cg,lrb_x301_500_cg,lrv_x301_500_cg,lrd_x301_500_cg,x,sigmoid)
error_500_const_cg301 = norm(vec(lyhat_x301_cg500.-y),2)/301
#write into the data file and plot
cw5cg301 = DataFrame(lrw_x301_500_cg,:auto)
CSV.write("lrw_x301_500_cg.csv",cw5cg301)
cb5cg301 = DataFrame([lrb_x301_500_cg],:auto)
CSV.write("lrb_x301_500_cg.csv",cb5cg301)
cv5cg301 = DataFrame(lrv_x301_500_cg,:auto)
CSV.write("lrv_x301_500_cg.csv",cv5cg301)
cd5cg301 = DataFrame([lrd_x301_500_cg],:auto)
CSV.write("lrd_x301_500_cg.csv",cd5cg301)
cy5cg301 = DataFrame(lyhat_x301_cg500,:auto)
CSV.write("lyhat_x301_CG_500.csv",cy5cg301)

using Plots
Plots.plot(x,lyhat_x301_cg500,label="NN_CG_500(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x301_500_cgls, lrb_x301_500_cgls, lrv_x301_500_cgls, lrd_x301_500_cgls = LM_1D_CGLS(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
end
lyhat_x301_cgls500 = one_hidden_layer_nn(lrw_x301_500_cgls,lrb_x301_500_cgls,lrv_x301_500_cgls,lrd_x301_500_cgls,x,sigmoid)
error_500_const_cgls301 = norm(vec(lyhat_x301_cgls500.-y),2)/301
#write into the data file and plot
cw5cgls301 = DataFrame(lrw_x301_500_cgls,:auto)
CSV.write("lrw_x301_500_cgls.csv",cw5cgls301)
cb5cgls301 = DataFrame([lrb_x301_500_cgls],:auto)
CSV.write("lrb_x301_500_cgls.csv",cb5cgls301)
cv5cgls301 = DataFrame(lrv_x301_500_cgls,:auto)
CSV.write("lrv_x301_500_cgls.csv",cv5cgls301)
cd5cgls301 = DataFrame([lrd_x301_500_cgls],:auto)
CSV.write("lrd_x301_500_cgls.csv",cd5cgls301)
cy5cgls301 = DataFrame(lyhat_x301_cgls500,:auto)
CSV.write("lyhat_x301_CGLS_500.csv",cy5cgls301)

using Plots
Plots.plot(x,lyhat_x301_cgls500,label="NN_CGLS_500(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)



#Test with x=LinRange(0,1,301) and identity being the activation function
#r=100
#CG
@time begin
    lrw_x301_100_cgi, lrb_x301_100_cgi, lrv_x301_100_cgi, lrd_x301_100_cgi = LM_1D_CG(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,identity)
end
lyhat_x301i_cg100 = one_hidden_layer_nn(lrw_x301_100_cgi,lrb_x301_100_cgi,lrv_x301_100_cgi,lrd_x301_100_cgi,x,identity)
error_100i_const_cg301 = norm(vec(lyhat_x301i_cg100.-y),2)/301
#write into the data file and plot
cw1cg301i = DataFrame(lrw_x301_100_cgi,:auto)
CSV.write("lrw_x301_100_cgi.csv",cw1cg301i)
cb1cg301i = DataFrame([lrb_x301_100_cgi],:auto)
CSV.write("lrb_x301_100_cgi.csv",cb1cg301i)
cv1cg301i = DataFrame(lrv_x301_100_cgi,:auto)
CSV.write("lrv_x301_100_cgi.csv",cv1cg301i)
cd1cg301i = DataFrame([lrd_x301_100_cgi],:auto)
CSV.write("lrd_x301_100_cgi.csv",cd1cg301i)
cy1cg301i = DataFrame(lyhat_x301i_cg100,:auto)
CSV.write("lyhat_x301i_CG_100.csv",cy1cg301i)

using Plots
Plots.plot(x,lyhat_x301i_cg100,label="NNi_CG_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x301_100_cglsi, lrb_x301_100_cglsi, lrv_x301_100_cglsi, lrd_x301_100_cglsi = LM_1D_CGLS(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,identity)
end
lyhat_x301i_cgls100 = one_hidden_layer_nn(lrw_x301_100_cglsi,lrb_x301_100_cglsi,lrv_x301_100_cglsi,lrd_x301_100_cglsi,x,identity)
error_100i_const_cgls301 = norm(vec(lyhat_x301i_cgls100.-y),2)/301
#write into the data file and plot
cw1cgls301i = DataFrame(lrw_x301_100_cglsi,:auto)
CSV.write("lrw_x301_100_cglsi.csv",cw1cgls301i)
cb1cgls301i = DataFrame([lrb_x301_100_cglsi],:auto)
CSV.write("lrb_x301_100_cglsi.csv",cb1cgls301i)
cv1cgls301i = DataFrame(lrv_x301_100_cglsi,:auto)
CSV.write("lrv_x301_100_cglsi.csv",cv1cgls301i)
cd1cgls301i = DataFrame([lrd_x301_100_cglsi],:auto)
CSV.write("lrd_x301_100_cglsi.csv",cd1cgls301i)
cy1cgls301i = DataFrame(lyhat_x301i_cgls100,:auto)
CSV.write("lyhat_x301i_CGLS_100.csv",cy1cgls301i)

using Plots
Plots.plot(x,lyhat_x301i_cgls100,label="NNi_CGLS_100(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#r=300
#CG
@time begin
    lrw_x301_300_cgi, lrb_x301_300_cgi, lrv_x301_300_cgi, lrd_x301_300_cgi = LM_1D_CG(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,identity)
end
lyhat_x301i_cg300 = one_hidden_layer_nn(lrw_x301_300_cgi,lrb_x301_300_cgi,lrv_x301_300_cgi,lrd_x301_300_cgi,x,sigmoid)
error_300i_const_cg301 = norm(vec(lyhat_x301i_cg300.-y),2)/301
#write into the data file and plot
cw3cg301i = DataFrame(lrw_x301_300_cgi,:auto)
CSV.write("lrw_x301_300_cgi.csv",cw3cg301i)
cb3cg301i = DataFrame([lrb_x301_300_cgi],:auto)
CSV.write("lrb_x301_300_cgi.csv",cb3cg301i)
cv3cg301i = DataFrame(lrv_x301_300_cgi,:auto)
CSV.write("lrv_x301_300_cgi.csv",cv3cg301i)
cd3cg301i = DataFrame([lrd_x301_300_cgi],:auto)
CSV.write("lrd_x301_300_cgi.csv",cd3cg301i)
cy3cg301i = DataFrame(lyhat_x301i_cg300,:auto)
CSV.write("lyhat_x301i_CG_300.csv",cy3cg301i)

using Plots
Plots.plot(x,lyhat_x301i_cg300,label="NNi_CG_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    lrw_x301_300_cglsi, lrb_x301_300_cglsi, lrv_x301_300_cglsi, lrd_x301_300_cglsi = LM_1D_CGLS(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,identity)
end
lyhat_x301i_cgls300 = one_hidden_layer_nn(lrw_x301_300_cglsi,lrb_x301_300_cglsi,lrv_x301_300_cglsi,lrd_x301_300_cglsi,x,identity)
error_300i_const_cgls301 = norm(vec(lyhat_x301i_cgls300.-y),2)/301
#write into the data file and plot
cw3cgls301i = DataFrame(lrw_x301_300_cglsi,:auto)
CSV.write("lrw_x301_300_cglsi.csv",cw3cgls301i)
cb3cgls301i = DataFrame([lrb_x301_300_cglsi],:auto)
CSV.write("lrb_x301_300_cglsi.csv",cb3cgls301i)
cv3cgls301i = DataFrame(lrv_x301_300_cglsi,:auto)
CSV.write("lrv_x301_300_cglsi.csv",cv3cgls301i)
cd3cgls301i = DataFrame([lrd_x301_300_cglsi],:auto)
CSV.write("lrd_x301_300_cglsi.csv",cd3cgls301i)
cy3cgls301i = DataFrame(lyhat_x301i_cgls300,:auto)
CSV.write("lyhat_x301i_CGLS_300.csv",cy3cgls301i)

using Plots
Plots.plot(x,lyhat_x301i_cgls300,label="NNi_CGLS_300(x)",seriestype=:scatter)
show()
Plots.plot!(x,y,label="real_solution",linewidth=2.0)