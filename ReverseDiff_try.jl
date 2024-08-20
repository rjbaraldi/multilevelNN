using ReverseDiff, BenchmarkTools
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

function one_hidden_layer_nn(
    input_weights,#input_weights is nodes_num*1
    input_biases,#input_biases is nodes_num*1
    output_weights,#output_weights is nodes_num*1
    output_bias,#output_weights is a scalar
    data,#sampled nD-PDEs variables matrix, 
    σ,#activation function,
    ) #where {T<: Real}
    inner =   input_weights * transpose(data)  .+ input_biases
    active = σ.(inner)
    output = transpose(output_weights) * active .+ output_bias
    return output'
end

#Try ForwardDiff
nn_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights ->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
nn_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
nn_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
nn_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_bias -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)

#Test Data
w_try = reshape(rand(3),(3,1))
b_try = ones(3)
v_try = reshape(rand(3),(3,1))
d_try = zeros(1)
x_try = collect(LinRange(0,1,5))
nn_1d_w(w_try,b_try,v_try,d_try,x_try,sigmoid) #works, 5×3 Matrix{Float64}
nn_1d_b(w_try,b_try,v_try,d_try,x_try,sigmoid) #works, 5×3 Matrix{Float64}
nn_1d_v(w_try,b_try,v_try,d_try,x_try,sigmoid) #works, 5×3 Matrix{Float64}
nn_1d_d(w_try,b_try,v_try,d_try,x_try,sigmoid) #works, 5×1 Matrix{Float64}
#Try ReverseDiff
nn_re_w(input_weights,input_biases,output_weights,output_bias,data,σ) = ReverseDiff.jacobian(input_weights ->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
nn_re_b(input_weights,input_biases,output_weights,output_bias,data,σ) = ReverseDiff.jacobian(input_biases -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
nn_re_v(input_weights,input_biases,output_weights,output_bias,data,σ) = ReverseDiff.jacobian(output_weights -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
nn_re_d(input_weights,input_biases,output_weights,output_bias,data,σ) = ReverseDiff.jacobian(output_bias -> one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data,σ),output_bias)
nn_re_w(w_try,b_try,v_try,d_try,x_try,sigmoid) #ERROR: DimensionMismatch: A has dimensions(3,5) but B has dimensions (1,5)
nn_re_b(w_try,b_try,v_try,d_try,x_try,sigmoid) #ERROR: BoundsError: attempt to access 5×3 Matrix{Float64} at index [1,1,2]
nn_re_v(w_try,b_try,v_try,d_try,x_try,sigmoid) #ERROR: BoundsError: attempt to access 5×3 Matrix{Float64} at index [1,1,2]
nn_re_d(w_try,b_try,v_try,d_try,x_try,sigmoid) #works, 5×1 Matrix{Float64}

