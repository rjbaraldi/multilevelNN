#In this file, we give the codes for MLM in 1D
using LeastSquaresOptim
using PyPlot
using DifferentialEquations
using Optim, Flux
using Random,Printf
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
using ForwardDiff, SparseArrays
using Optimization, ProgressMeter, OptimizationOptimJL
using Zygote, FluxOptTools, Statistics
using IterativeSolvers
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



function nn_lap1D_x_FD(input_weights,input_biases,output_weights,output_bias,data,σ)
    inner =   input_weights * transpose(data[2:end-1])  .+ input_biases
    output = buildLap1D(data)*σ(inner)'
    return output*output_weights
end 

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

function tk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s)
    N_D = size(data)[1]-2
    N_BC = 2
    λ_p = 0.1
    f1 = F_1(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ)
    j1 = J_1(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ)
    first_term = norm(f1,2)^2+(2*(j1'*f1)'*s)[1]+s'*j1'*j1*s
    f20 = F_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    f2e = F_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    j20 = J_2(input_weights,input_biases,output_weights,output_bias,data[1],σ)
    j2e = J_2(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    second_term = norm(f20,2)^2+(2*(j20'*f20)'*s)[1]+s'*j20'*j20*s+norm(f2e,2)^2+(2*(j2e'*f2e)'*s)[1]+s'*j2e'*j2e*s
    return first_term/(2*N_D)+λ_p*second_term/(2*N_BC)
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

function matrixA(input_weights,input_biases,output_weights,output_bias,data,σ)
    si = size(data)[1]-2
    λ_p = 0.1
    F_1w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_1b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_1v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    F_2w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_2b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_2v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    F_1_w = F_1w(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ)
    F_1_b = F_1b(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ)
    F_1_v = F_1v(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ)
    F_2_w = F_2w(input_weights,input_biases,output_weights,output_bias,[data[1],data[end]],σ)
    F_2_b = F_2b(input_weights,input_biases,output_weights,output_bias,[data[1],data[end]],σ)
    F_2_v = F_2v(input_weights,input_biases,output_weights,output_bias,[data[1],data[end]],σ)
    return F_1_w'*F_1_w/(norm(F_1_w,Inf)*si)+F_1_b'*F_1_b/(norm(F_1_b,Inf)*si)+F_1_v'*F_1_v/(norm(F_1_v,Inf)*si)+λ_p*F_2_w'*F_2_w/(norm(F_2_w,Inf)*2)+λ_p*F_2_b'*F_2_b/(norm(F_2_b,Inf)*2)+λ_p*F_2_v'*F_2_v/(norm(F_2_v,Inf)*2)
end


function mH_1D(wh,bh,vh,dh,data,σ,sH,λ,P,R)
    wH = P*wh
    bH = P*bh
    vH = P*vh
    dH = dh
    R_grad_w = R*obj_1d_w(wh,bh,vh,dh,data,σ)
    R_grad_b = R*obj_1d_b(wh,bh,vh,dh,data,σ)
    R_grad_v = R*obj_1d_v(wh,bh,vh,dh,data,σ)
    grad_obj_d = obj_1d_d(wh,bh,vh,dh,data,σ)
    R_grad = vcat(R_grad_w,R_grad_b,R_grad_v,grad_obj_d)
    first_term = mk_1d(wH,bH,vH,dH,data,σ,sH,λ)
    second_term =((R_grad-grad_obj_1d(wH,bH,vH,dH,data,σ))'*sH)[1]
    return first_term+second_term
end




function mH_line_A(wh,bh,vh,dh,data,σ,sH_size,λ,P,R)
   wH = P*wh
   bH = P*bh
   vH = P*vh
   dH = dh
   return line_mk_A(wH,bH,vH,dH,data,σ,sH_size,λ)
end


function mH_line_b(wh,bh,vh,dh,data,σ,sH_size,λ,P,R)
   wH = P*wh
   bH = P*bh
   vH = P*vh
   dH = dh
   R_grad_w = R*obj_1d_w(wh,bh,vh,dh,data,σ)
   R_grad_b = R*obj_1d_b(wh,bh,vh,dh,data,σ)
   R_grad_v = R*obj_1d_v(wh,bh,vh,dh,data,σ)
   grad_obj_d = obj_1d_d(wh,bh,vh,dh,data,σ)
   R_grad = vcat(R_grad_w,R_grad_b,R_grad_v,grad_obj_d)
   return line_mk_b(wH,bH,vH,dH,data,σ,sH_size,λ)+R_grad-grad_obj_1d(wH,bH,vH,dH,data,σ)
end


function MLM_1D_CG(input_weights,input_biases,output_weights,output_bias,data,σ,l)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    #θ = 1e-2
    λ_min = 1e-6
    ϵ = 1e-4
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
    A = matrixA(input_weights,input_biases,output_weights,output_bias,data,σ)
    splitting = splitting_A(A,ϵ_AMG)#splitting = AlgebraicMultigrid.RS_CF_splitting(sparse_S,sparse_T)
    n_c = count(!iszero,splitting)
    n_f = size(splitting)[1]
    C_set = findall(x -> x != 0,splitting)
    fP = zeros((n_f,n_c))
    #if i in C, x_F=x_c
    fP[C_set,1:n_c] .= I(n_c)
    fR = transpose(fP)#/norm(fP)
    #fP = transpose(fR)
    #σ_R = norm(fR)
    H_size = size(fP)[2]
    para_size = size(input_biases)[1]
    s_size = 3*para_size+1
    
    sH_size = 3*H_size+1
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        grad_obj = grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_w = obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_b = obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_v = obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_d = obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad_w = fR*grad_obj_w
        R_grad_b = fR*grad_obj_b
        R_grad_v = fR*grad_obj_v
        R_grad = vcat(R_grad_w,R_grad_b,R_grad_v,grad_obj_d)
        @show norm(R_grad_w,2)/(κ_H*norm(grad_obj_w,2)) norm(R_grad_b,2)/(κ_H*norm(grad_obj_b,2)) norm(R_grad_v,2)/(κ_H*norm(grad_obj_v,2))
        if l>1 && norm(R_grad_w,2) >= κ_H*norm(grad_obj_w,2) && norm(R_grad_b,2) >= κ_H*norm(grad_obj_b,2) && norm(R_grad_v,2) >= κ_H*norm(grad_obj_v,2) && norm(R_grad,2) > ϵ_H
            #pre_smoothing
            s = IterativeSolvers.gauss_seidel(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))
            input_weights = input_weights.+s[1:para_size]
            input_biases = input_biases.+s[para_size+1:2*para_size]
            output_weights = output_weights.+s[2*para_size+1:3*para_size]
            output_bias = output_bias.+s[end]
            mH(s_H) = mH_1D(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR,fR)
            s_H = Krylov.cg(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR,fR)))[1]
            #s_H = IterativeSolvers.gauss_seidel(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR,fR)))
            s = vcat(fP*s_H[1:H_size],fP*s_H[H_size+1:2*H_size],fP*s_H[2*H_size+1:3*H_size],s_H[end])
            mh(s_H) = mH(s_H)#/σ_R
            #fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
            #@show fk(s) fk(zeros(s_size))
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = mh(zeros(3*H_size+1))-mh(s_H)
            ρ = ρkn/ρkd
            @show ρkn ρkd ρ
        else
            mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
            s = Krylov.cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
            #@show cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2:end]
        #@show s
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = mk(zeros(s_size))-mk(s)
            ρ = ρkn/ρkd
            @show ρkn ρkd ρ
        end
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
    return input_weights,input_biases,output_weights,output_bias
end


function MLM_1D_CGLS(input_weights,input_biases,output_weights,output_bias,data,σ,l)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    #θ = 1e-2
    λ_min = 1e-6
    ϵ = 1e-4
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
    A = matrixA(input_weights,input_biases,output_weights,output_bias,data,σ)
    splitting = splitting_A(A,ϵ_AMG)#splitting = AlgebraicMultigrid.RS_CF_splitting(sparse_S,sparse_T)
    n_c = count(!iszero,splitting)
    n_f = size(splitting)[1]
    C_set = findall(x -> x != 0,splitting)
    fP = zeros((n_f,n_c))
    #if i in C, x_F=x_c
    fP[C_set,1:n_c] .= I(n_c)
    fR = transpose(fP)#/norm(fP)
    #fP = transpose(fR)
    #σ_R = norm(fR)
    H_size = size(fP)[2]
    para_size = size(input_biases)[1]
    s_size = 3*para_size+1
    
    sH_size = 3*H_size+1
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        grad_obj = grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_w = obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_b = obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_v = obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj_d = obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad_w = fR*obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad_b = fR*obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad_v = fR*obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad = vcat(R_grad_w,R_grad_b,R_grad_v,grad_obj_d)
        @show norm(R_grad_w,2)/(κ_H*norm(grad_obj_w,2)) norm(R_grad_b,2)/(κ_H*norm(grad_obj_b,2)) norm(R_grad_v,2)/(κ_H*norm(grad_obj_v,2))
        if l>1 && norm(R_grad_w,2) >= κ_H*norm(grad_obj_w,2) && norm(R_grad_b,2) >= κ_H*norm(grad_obj_b,2) && norm(R_grad_v,2) >= κ_H*norm(grad_obj_v,2) && norm(R_grad,2) > ϵ_H
            s = IterativeSolvers.gauss_seidel(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))
            input_weights = input_weights.+s[1:para_size]
            input_biases = input_biases.+s[para_size+1:2*para_size]
            output_weights = output_weights.+s[2*para_size+1:3*para_size]
            output_bias = output_bias.+s[end]
            mH(s_H,λ) = mH_1D(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR,fR)
            optprob = OptimizationFunction(mH,Optimization.AutoForwardDiff())
            s_H = zeros(sH_size)
            prob = Optimization.OptimizationProblem(optprob,s_H,λ)
            s_H = solve(prob,Optim.LBFGS())
            #s_H = Krylov.cgls(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR)))[1]
            #@show cgls(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR)))[2:end]
            s = vcat(fP*s_H[1:H_size],fP*s_H[H_size+1:2*H_size],fP*s_H[2*H_size+1:3*H_size],s_H[end])
            mh(s_H) = mH(s_H,λ)#/σ_R
            #fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
            #@show fk(s) fk(zeros(s_size))
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = mh(zeros(3*H_size+1))-mh(s_H)
            ρ = ρkn/ρkd
            @show ρkn ρkd ρ
        else
            mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
            s = cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
            #@show cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2:end]
        #@show s
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = mk(zeros(s_size))-mk(s)
            ρ = ρkn/ρkd
            @show ρkn ρkd ρ
        end
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
    return input_weights,input_biases,output_weights,output_bias
end

x = collect(LinRange(0,1,41))
yreal = -x.^2/2

#r = 100 with σ=sigmoid, x=LinRange(0,1,41)
using CSV, DataFrames
rw_lu_100 = reshape(CSV.read("rw_lu_100.csv",DataFrame)[!,1],(100,1))
rb_lu_100 = CSV.read("rb_lu_100.csv",DataFrame)[!,1]
rv_lu_100 = reshape(CSV.read("rv_lu_100.csv",DataFrame)[!,1],(100,1))
rd_lu_100 = CSV.read("rd_lu_100.csv",DataFrame)[!,1]

@time begin
    nwm100cg,nbm100cg,nvm100cg,ndm100cg = MLM_1D_CG(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid,2)
end

y_pred_100cg = one_hidden_layer_nn(nwm100cg,nbm100cg,nvm100cg,ndm100cg,x,sigmoid)
error_100cg = norm(y_pred_100cg-yreal,2)/41
    

#r = 300 with σ=sigmoid, x=LinRange(0,1,41)
rw_lu_300 = reshape(CSV.read("rw_lu_300.csv",DataFrame)[!,1],(300,1))
rb_lu_300 = CSV.read("rb_lu_300.csv",DataFrame)[!,1]
rv_lu_300 = reshape(CSV.read("rv_lu_300.csv",DataFrame)[!,1],(300,1))
rd_lu_300 = CSV.read("rd_lu_300.csv",DataFrame)[!,1]

@time begin
    nwm300cg,nbm300cg,nvm300cg,ndm300cg = MLM_1D_CG(rw_lu_300,rb_lu_300,rv_lu_300,rd_lu_300,x,sigmoid,2)
end
y_pred_300cg = one_hidden_layer_nn(nwm300cg,nbm300cg,nvm300cg,ndm300cg,x,sigmoid)
error_300cg = norm(y_pred_300cg-yreal,2)/41
   

#r=500
rw_lu_500 = reshape(CSV.read("rw_lu_500.csv",DataFrame)[!,1],(500,1))
rb_lu_500 = CSV.read("rb_lu_500.csv",DataFrame)[!,1]
rv_lu_500 = reshape(CSV.read("rv_lu_500.csv",DataFrame)[!,1],(500,1))
rd_lu_500 = CSV.read("rd_lu_500.csv",DataFrame)[!,1]
@time begin
    nwm500cg,nbm500cg,nvm500cg,ndm500cg = MLM_1D_CG(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid,2) 
end
y_pred_500cg = one_hidden_layer_nn(nwm500cg,nbm500cg,nvm500cg,ndm500cg,x,sigmoid)
error_500cg = norm(y_pred_500cg-yreal,2)/41



