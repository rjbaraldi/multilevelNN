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


#function one_hidden_layer_nn_2d(
#    input_weights,#input_weights is nodes_num*n
#    input_biases,#input_biases is nodes_num*1
#    output_weights,#output_weights is nodes_num*1
#    output_bias,#output_weights is a scalar
#    data,#sampled nD-PDEs variables matrix, 
#    σ#activation function,
#    )
#    sample_x_size,sample_y_size = size(data)
#    nodes_num = size(output_weights)[1]
#    output_2d = zeros(sample_x_size,sample_y_size)
#    inner_2d = zeros(sample_x_size,sample_y_size)
#    for i in 1:sample_x_size
#        for j in 1:sample_y_size
#            for k in 1:nodes_num
#                for t in 1:2
#                inner_2d[i,j] += input_weights[k,t]*data[i,j][t]
#                end
#                inner_2d[i,j] = inner_2d[i,j].+input_biases[k]
#                output_2d[i,j] += output_weights[k]*σ(inner_2d[i,j]) 
#            end
#        end
#    end
#    return  output_2d.+output_bias
#end


function possion_2d(real_solution::Function)
    #@parameters x,y
    #@variables u(..)
    u(x,y) = real_solution(x,y)
    u_x(x,y) = ForwardDiff.derivative(x -> u(x,y),x)
    u_xx(x,y) = ForwardDiff.derivative(x -> u_x(x,y),x)
    u_y(x,y) = ForwardDiff.derivative(y -> u(x,y),y)
    u_yy(x,y) = ForwardDiff.derivative(y-> u_y(x,y),y)
    g_1(x,y) = -(u_xx(x,y)+u_yy(x,y))
    g_2(x,y) = real_solution(x,y)
    return u,g_1,g_2
end

function cos_v_2d(ν)
    function cos_v_xy(x,y)
        return cos(ν*(x+y))
    end
    return cos_v_xy
end

function constant_2d(a)
    function a_func_2d(x,y)
        return (-a/4)*x^2+(-a/4)*y^2
    end
    return a_func_2d
end
   

function data_2d(xx::AbstractVector,yy::AbstractVector)
    row_size = size(xx)[1]
    col_size = size(yy)[1]
    A = Array{Tuple{Float64,Float64},2}(undef,row_size,col_size)
    for i in 1:row_size
        for j in 1:col_size
         A[i,j] = (xx[i],yy[j])
        end
    end
    return collect.(A)
 end



function data_2d_vector(x,y)
   A_M = data_2d(x,y)
   nx=length(x)
   ny=length(y)
   A_M = reshape(A_M',:,1)
   A_out = zeros(2,nx*ny)
   for i in 1:nx*ny
    A_out[1,i] = A_M[i,1][1]
    A_out[2,i] = A_M[i,1][2]
   end
   return A_out'
end

#A_trytry=data_2d_vector(xx_try,yy_try)

#one_hidden_layer_nn_2ds(ww_try,bb_try,vv_try,dd_try,A_trytry,identity)


#grad_x_nn(ww_1_try,bb_try,vv_try,dd_try,xx_try,identity)


function buildLap2D(x,y)
    nx = length(x)
    ny = length(y)

    D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
    Id = I(nx)
    A = kron(D, Id) + kron(Id, D)
    A /= (nx+1)^2
    return A
end

#function grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
#    grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)= ForwardDiff.jacobian(data->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data, σ),data)
#    size_m = size(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ))[1]
#    grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)[:,1:size_m])
#    grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)[:,size_m+1:end])
#    return grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
#end

#function hess_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
#    size_m = size(data)[1]
    #grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)=grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
    #grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)= ForwardDiff.jacobian(data->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data, σ),data)
    #grad_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = vcat(grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ))
#    hess_x_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(ForwardDiff.jacobian(data->grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)[1],data))
#    hess_y_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(ForwardDiff.jacobian(data->grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)[2],data)[:,size_m+1:end])
    
    #@show hess_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)   
#    return hess_x_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) + hess_y_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)
#end


#separate the data into interior and boundary points
function separate_ib(data)
    data_size = size(data)[1]
    interior_data = []
    boundary_data = []
    for  j in 1:data_size
        if data[j,1] == data[:,1][1] || data[j,1]==data[:,1][end] ||data[j,2]==data[:,2][1]||data[j,2]==data[:,2][end]
                push!(boundary_data,[data[j,1],data[j,2]])
        else
                push!(interior_data,[data[j,1],data[j,2]])
        end
    end
    interior_data = reshape(collect(Iterators.flatten(interior_data)),length(interior_data[1]),length(interior_data))'
    boundary_data = reshape(collect(Iterators.flatten(boundary_data)),length(boundary_data[1]),length(boundary_data))'
    return interior_data, boundary_data
end

#function obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ)
#    g_1 = possion_2d(cos_v_2d(20))[2] #the right-hand side could choose other PDEs
#    g_2 = possion_2d(cos_v_2d(20))[3]
#    loss = 0
#    penalty = 0
#    data_size = size(data)[1]
    #separate the data into interior and boundary points
#    interior_data, boundary_data=separate_ib(data)
#    N_D = size(interior_data)[1]
#    N_BC = size(boundary_data)[1]
#    for k in 1:N_D
#        loss += norm(g_1(interior_data[k,1],interior_data[k,2])+hess_nn_2d(input_weights,input_biases,output_weights,output_bias,interior_data,σ)[k],2)^2
#    end
#    λ_p = 0.1*sqrt(data_size)
#    for l in 1:N_BC
#        penalty += norm(g_2(boundary_data[l,1],boundary_data[l,2])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,boundary_data,σ)[l],2)^2
#    end
#    loss = loss/(2*N_D)
    #@show whole
#    penalty = λ_p*(penalty)/(2*N_BC)
#    return penalty + loss
#end

#inter = separate_ib(data_data)[1]
#inter = reshape(collect(Iterators.flatten(inter)),length(inter[1]),length(inter))'
#data_data
#hess_nn_2d(ww_try,bb_try,vv_try,dd_try,inter,sigmoid)
#obj_2d(ww_try,bb_try,vv_try,dd_try,data_data,sigmoid)

function y_training_data(x1,x2,real_func)
    size_x1 = size(x1)[1]
    y_training_data = zeros(size_x1,size_x1)
    for i in 1:size_x1
        for j in 1:size_x1
            y_training_data[i,j] = real_func(x1[i],x2[j])
        end
    end
    y_training_data = reshape(y_training_data',:,1)
    return y_training_data
end


function nn_initial(num_nodes_list,x1::AbstractVector,x2::AbstractVector,real_func::Function)
    y_data = y_training_data(x1,x2,real_func)
    y_real = zeros(size(y_data)[1])
    for l in 1:size(y_real)[1]
    y_real[l] = y_data[l]
    end
    x_data = [[x1[i],x2[j]] for i in 1:size(x1)[1] for j in 1:size(x2)[1]]
    data_training = [(x_data[i],[y_data[i]]) for i in 1:size(y_data)[1]]
    best_model = nothing
    best_loss = Inf
    for n in num_nodes_list
    Layer_hidden = Flux.Dense(2=>n,sigmoid)
    Layer_output = Flux.Dense(n=>1,identity)
    Model = Flux.Chain(Layer_hidden,Layer_output)
    learning_rate = 0.001
    opt = Flux.setup(Flux.Adam(learning_rate),Model)
    Flux.train!(Model,data_training,opt) do m,x,y
        sum(abs.(m(x).-y).^2)/length(x_data)^2
    end
    y_pred = zeros(size(y_data)[1])
    for k in 1:size(y_data)[1]
        y_pred[k] = Model.(x_data)[k][1]
    end
    loss = norm(y_real-y_pred,2)/size(y_data)[1]
    @show loss
    if loss < best_loss
        best_loss = loss
        best_model = Model
    else
        best_loss = best_loss
        best_model = best_model
    end
    end
    return best_loss,best_model
end

Layer_hidden = Flux.Dense(2=>200,sigmoid)#the number of nodes from the result of above function
Layer_output = Flux.Dense(200=>1,identity)
Model = Flux.Chain(Layer_hidden,Layer_output)
x1_model = collect(LinRange(0,1,41))
x2_model = x1_model
x_model = data_2d_vector(x1_model,x2_model)'
y_data = y_training_data(x1_model,x2_model,constant_2d(1))
#data_training = [(x_data[i],[y_data[i]]) for i in 1:size(y_data)[1]]
learning_rate = 0.001
opt = Flux.setup(Flux.Adam(learning_rate),Model)
y_model = zeros(1,size(y_data)[1])
for i in 1:size(y_data)[1]
    y_model[1,i] = y_data[i]
end
y_model
losses_model = []
y_hat = zeros(size(y_model)[2])
@info "epoch    loss"
@showprogress for epoch in 1:1000
    loss, grads = Flux.withgradient(Model) do m
        y_hat = m(x_model)
        Flux.Losses.mse(y_hat,y_model)
    end
    Flux.update!(opt,Model,grads[1])
    push!(losses_model,loss)
    (epoch %200 ==0) && @info @sprintf "%d  %8.3e" epoch losses_model[end] 
end  
#Flux.train!(Model,data_training,opt) do m,x,y
#    sum(abs.(m(x).-y).^2)/length(x_data)^2
#end
fw2d = Layer_hidden.weight
fb2d = Layer_hidden.bias
fv2d = vec(Layer_output.weight)
fd2d = Layer_output.bias
input_weight_2d = Layer_hidden.weight.+rand(1)
input_bias_2d = Layer_hidden.bias.+rand(1)
output_weight_2d = vec(Layer_output.weight).+rand(1)
output_bias_2d = Layer_output.bias.+rand(1)

#flux_approx = one_hidden_layer_nn(input_weight_2d,input_bias_2d,output_weight_2d,output_bias_2d,data_2d_vector(x1_model,x2_model),sigmoid)

#obj_2d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_weights -> obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ), input_weights)
#obj_2d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(input_bias -> obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ), input_bias)
#obj_2d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_weights -> obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ), output_weights)
#obj_2d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.gradient(output_bias -> obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ), output_bias)


#approximate laplacian operator 
function obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    g_1 = possion_2d(constant_2d(1))[2] #the right-hand side could choose other PDEs
    g_2 = possion_2d(constant_2d(1))[3]
    loss = 0
    penalty = 0
    size_x1 = size(x1)[1]
    size_x2 = size(x2)[1]
    inter_data = data_2d_vector(x1[2:end-1],x2[2:end-1])
    g_1_matrix = zeros((size_x1-2),(size_x2-2))
    for i in 2:size_x1-1
        for j in 2:size_x1-1
            g_1_matrix[i-1,j-1] = g_1(x1[i],x2[j]) 
        end
    end
    g_1_vector = reshape(g_1_matrix',:,1)
    loss =norm(g_1_vector+buildLap2D(x1[2:end-1],x2[2:end-1])*one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,inter_data,σ),2)^2
    λ_p = 0.1*size_x1
    for k in 1:size_x1
        penalty += norm(g_2(x1[1],x2[k])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,[x1[1] x2[k]],σ)[1,1],2)^2+norm(g_2(x1[end],x2[k])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,[x1[end] x2[k]],σ)[1,1],2)^2+norm(g_2(x1[k],x2[1])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,[x1[k] x2[1]],σ)[1,1],2)^2+norm(g_2(x1[k],x2[end])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,[x1[k] x2[end]],σ)[1,1],2)^2
    end
    loss = loss/(2*(size_x1-2)*(size_x2-2))
    #@show whole
    penalty = λ_p*(penalty)/(4*(size_x1+size_x2-2))
    return penalty + loss
end

#obj_2d_approx(input_weight_2d,input_bias_2d,output_weight_2d,output_bias_2d,x1_model,x2_model,sigmoid)

obj_2d_approx_w(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = ForwardDiff.gradient(input_weights -> obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ), input_weights)
obj_2d_approx_b(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = ForwardDiff.gradient(input_biases -> obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ), input_biases)
obj_2d_approx_v(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = ForwardDiff.gradient(output_weights -> obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ), output_weights)
obj_2d_approx_d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = ForwardDiff.gradient(output_bias -> obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ), output_bias)
obj_2d_approx_w1(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = obj_2d_approx_w(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)[:,1]
obj_2d_approx_w2(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = obj_2d_approx_w(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)[:,2]
grad_obj_2d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ) = vcat(obj_2d_approx_w1(input_weights,input_biases,output_weights,output_bias,x1,x2,σ),obj_2d_approx_w2(input_weights,input_biases,output_weights,output_bias,x1,x2,σ),obj_2d_approx_b(input_weights,input_biases,output_weights,output_bias,x1,x2,σ),obj_2d_approx_v(input_weights,input_biases,output_weights,output_bias,x1,x2,σ),obj_2d_approx_d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ))


function MLM_2D(input_weights,input_biases,output_weights,output_bias,x1,x2,σ,l)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    θ = 1e-2
    λ_min = 1e-6
    ϵ = 1e-3
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
    A_AMG = matrix_A_2d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    P = prolongation(A_AMG, ϵ_AMG)
    R = transpose(P)
    H_size = size(P)[2]
    para_size = size(input_biases)[1]
    s_size = 4*para_size+1
    while norm(grad_obj_2d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ),2) > ϵ
       g_w1 = obj_2d_approx_w1(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
       g_w2 = obj_2d_approx_w2(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
       g_b = obj_2d_approx_b(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
       g_v = obj_2d_approx_v(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
       g_d = obj_2d_approx_d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
       g_withoutd = vcat(g_w1,g_w2,g_b,g_v)
       grad_obj = vcat(g_w1,g_w2,g_b,g_v,g_d)
       R_block = vcat(hcat(R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2])),hcat(zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2])),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2])),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R))
       R_grad = R_block*g_withoutd
       R_grad_including_d = vcat(R_grad,g_d)
       if l>1 && norm(R_grad_including_d,2) >= κ_H*norm(grad_obj,2) && norm(R_grad,2) > ϵ_H 
        w_H = R*input_weights
        b_H = R*input_biases
        v_H = R*output_weights
        d_H = output_bias
        f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = obj_2d_approx(hcat(w_H[:,1].+s_H[1:H_size],w_H[:,2].+s_H[H_size+1:2*H_size]),b_H.+s_H[2*H_size+1:3*H_size],v_H.+s_H[3*H_size+1:end-1],d_H.+s_H[end],x1,x2,σ)
        grad_H_w(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = ForwardDiff.gradient(w_H -> f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ), w_H)
        grad_H_b(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = ForwardDiff.gradient(b_H -> f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ), b_H)
        grad_H_v(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = ForwardDiff.gradient(v_H -> f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ), v_H)
        grad_H_d(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = ForwardDiff.gradient(d_H -> f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ), d_H)
        grad_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ) = vcat(reshape(grad_H_w(w_H,b_H,v_H,d_H,s_H,x1,x2,σ)',:,1),grad_H_b(w_H,b_H,v_H,d_H,s_H,x1,x2,σ),grad_H_v(w_H,b_H,v_H,d_H,s_H,x1,x2,σ),grad_H_d(w_H,b_H,v_H,d_H,s_H,x1,x2,σ))
        m_H(s_H) = f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ)+(R_grad_including_d-vec(grad_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ)))'*s_H +(λ*norm(s_H,2)^2)/2
        s_H_0_size = 4*size(b_H)[1]+1
        s_H_0 = 0.1*ones(s_H_0_size)
        s_H = Optim.minimizer(optimize(m_H, s_H_0))
        s_h = zeros(s_size)
        P_block = vcat(hcat(P,zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2])),hcat(zeros(size(P)[1],size(P)[2]),P,zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2])),hcat(zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),P,zeros(size(P)[1],size(P)[2])),hcat(zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),P))
        s_h[1:end-1] .= P_block*(s_H[1:end-1])
        s_h[end] = s_H[end]
        m_k(s_H) = f_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ)+(R_grad_including_d-vec(grad_H(w_H,b_H,v_H,d_H,s_H,x1,x2,σ)))'*s_H 
        ρ_numerator = obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)-obj_2d_approx(hcat(input_weights[:,1].+s_h[1:para_size],input_weights[:,2].+s_h[para_size+1:2*para_size]),input_biases.+s_h[2*para_size+1:3*para_size],output_weights.+s_h[3*para_size+1:4*para_size],output_bias.+s_h[end],x1,x2,σ)
        ρ_denominator = m_k(zeros(s_H_0_size))-m_k(s_H)
        ρ = ρ_numerator/ρ_denominator
        @show ρ_numerator ρ_denominator
        if ρ >= η1
            input_weights[:,1] = input_weights[:,1].+s_h[1:para_size]
            input_weights[:,2] = input_weights[:,2].+s_h[para_size+1:2*para_size]
            input_biases = input_biases.+s_h[2*para_size+1:3*para_size]
            output_weights = output_weights.+s_h[3*para_size+1:4*para_size]
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
        #return input_weights,input_biases,output_weights,output_bias
       else
        m_h(s_h) = obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)+grad_obj'*s_h+s_h'*grad_obj*grad_obj'*s_h/2+(λ*norm(s_h,2)^2)/2
        s_h_size = 4*para_size+1
        s_h_0 = 0.1*ones(s_h_size)
        s_h = Optim.minimizer(optimize(m_h,s_h_0))
        ρ_numerator = obj_2d_approx(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)-obj_2d_approx(hcat(input_weights[:,1].+s_h[1:para_size],input_weights[:,2].+s_h[para_size+1:2*para_size]),input_biases.+s_h[2*para_size+1:3*para_size],output_weights.+s_h[3*para_size+1:4*para_size],output_bias.+s_h[end],x1,x2,σ)
        ρ_denominator = m_h(zeros(s_h_size))-m_h(s_h)
        ρ = ρ_numerator/ρ_denominator
        if ρ >= η1
            input_weights[:,1] = input_weights[:,1].+s_h[1:para_size]
            input_weights[:,2] = input_weights[:,2].+s_h[para_size+1:2*para_size]
            input_biases = input_biases.+s_h[2*para_size+1:3*para_size]
            output_weights = output_weights.+s_h[3*para_size+1:4*para_size]
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
        #return input_weights,input_biases,output_weights,output_bias
       end
    end
    return input_weights,input_biases,output_weights,output_bias
end
