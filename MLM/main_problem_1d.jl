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
using Optimization
using Zygote, FluxOptTools, Statistics

##Build neural network
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



#flux_1d_approx = one_hidden_layer_nn(fa_w,fa_b,fa_v,fa_d,x',sigmoid)




#Set objective function

#Set PDEs
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


#Gradient/Hessian of neural network with respect to data (the variable of pde)
#function grad_x_nn_1d(input_weights,input_biases,output_weights,output_bias,data, σ)
 #   grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)= ForwardDiff.jacobian(data->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data, σ),data)
  #  return diag(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ))
#end

#function hess_x_nn_1d(input_weights,input_biases,output_weights,output_bias,data, σ)
   # hess_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) = ForwardDiff.jacobian(data->grad_x_nn_1d(input_weights,input_biases,output_weights,output_bias,data, σ),data)
    #@show hess_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)   
    #return diag(hess_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ))
#end

#approximate laplacian operator
function buildLap1D(x)
    nx = length(x)
    D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
    D /= (nx-1)^2
    return D
end

#function obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
#    g_1 = possion_1d(constant(1))[2] #the right-hand side could choose other PDEs
#    g_2 = possion_1d(constant(1))[3]
 #   N_u = size(data)[1]-2
 #   loss = 0
 #   for i in 1:N_u
 #   loss += norm(g_1(data[i+1])+hess_x_nn_1d(input_weights,input_biases,output_weights,output_bias,data,σ)[i+1],2)^2
 #   end
 #   loss = loss/(2*N_u)
 #   λ_p = 0.1*size(data)[1]
 #   penalty = norm(g_2(data[1])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[1],2)^2+norm(g_2(data[end])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[end],2)^2
 #   penalty = λ_p*penalty/4
 #   return loss+penalty
#end


function obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ)
    g_1 = possion_1d(constant(1))[2] #the right-hand side could choose other PDEs
    g_2 = possion_1d(constant(1))[3]
    N_u = size(data)[1]-2
    g_1_data = g_1.(data[2:end-1])
    loss = norm(g_1_data+buildLap1D(data[2:end-1])*one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data[2:end-1],σ),2)^2
    loss = loss/(2*N_u)
    λ_p = 0.1*size(data)[1]
    penalty = norm(g_2(data[1])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[1],2)^2+norm(g_2(data[end])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ)[end],2)^2
    penalty = λ_p*penalty/4
    return loss+penalty
end

#function square(x)
#    return x.^2
#end
obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_weights->obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_biases->obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_weights->obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
obj_1d_approx_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_bias->obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)
grad_obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ) = vcat(obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ),obj_1d_approx_d(input_weights,input_biases,output_weights,output_bias,data,σ))
#Define the gradient of objective function with respect to the parameters of neural networks
#grad_obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_weights->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
#grad_obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_biases->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
#grad_obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_weights->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
#grad_obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_bias->obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)

#grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) = vcat(grad_obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ),grad_obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ),grad_obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ),grad_obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ))
#hessian_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)=grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)*grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)'
#hessian_obj_1d(ww_1_try,bb_try,vv_try,dd_try,x_x_x,identity)
#function taylor_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s)
#    #where s is in (3*nodes_num+1)*1
#    first_term = obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
#    second_term = grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)'*s
#    third_term = s'*hessian_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)*s/2
#    return first_term+second_term+third_term
#end
#function grad_taylor_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s)
#    return grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)+hessian_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)*s
#end

#function cgls(A, b, max_iteration, tol)
#    m, n = size(A)
#    x = zeros(n)  # Initial guess
#    r = b - A * x
#    p = r
#    rsold = dot(r, r)
#    for iter = 1:max_iteration
#        Ap = A * p
#        alpha = rsold / dot(Ap, Ap)
 #       x += alpha * p
 #       r -= alpha * Ap
  #      rsnew = dot(r, r)
   #     if sqrt(rsnew) < tol
   #         println("Converged in $iter iterations.")
    #        return x
     #   end
     #   p = r + (rsnew / rsold) * p
     #   rsold = rsnew
    #end
    #println("Maximum iterations reached without convergence.")
    #return x
#end

#give a suitable initial guess of the parameters in neural network


function LM_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    θ = 1e-2
    λ_min = 1e-6
    ϵ = 1e-4
    s_size = 3*size(output_weights)[1]+1
    para_size = size(output_weights)[1]
    #give an initial step sk
    s = 0.01*ones(s_size)
    while norm(grad_obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        g_w = obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_b = obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_v = obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_d = obj_1d_approx_d(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj = vcat(g_w,g_b,g_v,g_d)
        f = obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ)
        @show f
        m_h(s_h) = f+(grad_obj'*s_h)[1]+(s_h'*grad_obj*grad_obj'*s_h/2)[1]+(λ*norm(s_h,2)^2)/2
        m_s(s_h) = ForwardDiff.gradient(s_h->m_h(s_h),s_h)
        tolence_con = norm(m_s(s)+λ*s,2)/norm(s,2)
        @show tolence_con
        while tolence_con > θ
            s = Optim.minimizer(optimize(m_h, s)) #we can choose different values for the last two parameters 
        end
        s_w = s[1:para_size]
        s_b = s[para_size+1:2*para_size]
        s_v = s[2*para_size+1:3*para_size]
        s_d = s[end]
        @show s_d
        ρ_numerator = obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ)-obj_1d_approx(input_weights.+s_w,input_biases.+s_b,output_weights.+s_v,output_bias.+s_d,data,σ)
        ρ_denominator = m_h(zeros(s_size))-m_h(s)
        ρ = ρ_numerator/ρ_denominator
        @show ρ_numerator ρ_denominator
        if ρ >= η1
            input_weights = input_weights.+s_w
            input_biases = input_biases.+s_b
            output_weights = output_weights.+s_v
            output_bias = output_bias.+s_d
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



#give a suitable guess of the parameters in neural network, which can be used as the input of LM_1d
Layer_1 = Flux.Dense(1=>50, sigmoid)
output_layer = Flux.Dense(50=>1)
m = Flux.Chain(Layer_1, output_layer)
x = LinRange(0,1,41)'
y = -(x.^2)/2
loss() = mean(abs2,m(x) .- y)
pars = Flux.params(m)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
fa_w = Layer_1.weight
fa_b = Layer_1.bias
fa_v = vec(output_layer.weight)
fa_d = output_layer.bias
input_weights_0 = Layer_1.weight.+rand(1)
input_bias_0 = Layer_1.bias.+rand(1)
output_weight_0 = vec(output_layer.weight).+rand(1)
output_bias_0 = output_layer.bias.+rand(1)


function MLM_1d(input_weights,input_biases,output_weights,output_bias,data,σ,l)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    θ = 1e-2
    λ_min = 1e-6
    ϵ = 1e-4
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
    A_AMG = matrix_A_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
    P = prolongation(A_AMG, ϵ_AMG)
    R = transpose(P)
    R = R/sqrt(det(R*R'))
    H_size = size(P)[2]
    para_size = size(input_biases)[1]
    s_size = 3*para_size+1
    while norm(grad_obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
       g_w = obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ)
       g_b = obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ)
       g_v = obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ)
       g_d = obj_1d_approx_d(input_weights,input_biases,output_weights,output_bias,data,σ)
       g_withoutd = vcat(g_w,g_b,g_v)
       grad_obj = vcat(g_w,g_b,g_v,g_d)
       R_block = vcat(hcat(R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2])),hcat(zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2])),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R))
       R_grad = R_block*g_withoutd
       R_grad_including_d = vcat(R_grad,g_d)
       if l>1 && norm(R_grad_including_d,2) >= κ_H*norm(grad_obj,2) && norm(R_grad,2) > ϵ_H
        w_H = R*input_weights
        b_H = R*input_biases
        v_H = R*output_weights
        d_H = output_bias
        f_H(w_H,b_H,v_H,d_H,s_H,data,σ) = obj_1d_approx(w_H.+s_H[1:H_size],b_H.+s_H[H_size+1:2*H_size],v_H.+s_H[2*H_size+1:end-1],d_H.+s_H[end],data,σ)
        grad_H_w(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(w_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), w_H)
        grad_H_b(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(b_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), b_H)
        grad_H_v(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(v_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), v_H)
        grad_H_d(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(d_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), d_H)
        grad_H(w_H,b_H,v_H,d_H,s_H,data,σ) = vcat(grad_H_w(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_b(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_v(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_d(w_H,b_H,v_H,d_H,s_H,data,σ))
        m_H(s_H) = f_H(w_H,b_H,v_H,d_H,s_H,data,σ)+((R_grad_including_d-grad_H(w_H,b_H,v_H,d_H,s_H,data,σ))'*s_H)[1] +(λ*norm(s_H,2)^2)/2
        s_H_0_size = 3*size(b_H)[1]+1
        s_H_0 = 0.1*ones(s_H_0_size)
        s_H = Optim.minimizer(optimize(m_H, s_H_0))
        s_h = zeros(s_size)
        P_block = vcat(hcat(P,zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2])),hcat(zeros(size(P)[1],size(P)[2]),P,zeros(size(P)[1],size(P)[2])),hcat(zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),P))
        s_h[1:end-1] .= P_block*(s_H[1:end-1])
        s_h[end] = s_H[end]
        @show s_h[end]
        ρ_numerator = obj_1d_approx(input_weights,input_biases,output_weights,output_bias,data,σ)-obj_1d_approx(input_weights.+s_h[1:para_size],input_biases.+s_h[para_size+1:2*para_size],output_weights.+s_h[2*para_size+1:3*para_size],output_bias.+s_h[end],data,σ)
        ρ_denominator = m_H(zeros(s_H_0_size))-m_H(s_H)
        ρ = ρ_numerator/ρ_denominator
        @show ρ_numerator ρ_denominator ρ
        if ρ >= η1
            input_weights = input_weights.+s_h[1:para_size]
            input_biases = input_biases.+s_h[para_size+1:2*para_size]
            output_weights = output_weights.+s_h[2*para_size+1:3*para_size]
            output_bias = output_bias.+s_h[end]
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
        @show λ
       else
        LM_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
       end
    end
    return input_weights,input_biases,output_weights,output_bias
end

MLM_1d(input_weights_0,input_bias_0,output_weight_0,output_bias_0,x',sigmoid,2)
