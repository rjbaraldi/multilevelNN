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


function one_hidden_layer_nn_2d(
    input_weights,#input_weights is nodes_num*n
    input_biases,#input_biases is nodes_num*1
    output_weights,#output_weights is nodes_num*1
    output_bias,#output_weights is a scalar
    data,#sampled nD-PDEs variables matrix, 
    σ#activation function,
    )
    sample_x_size,sample_y_size = size(data)
    nodes_num = size(output_weights)[1]
    output_2d = zeros(sample_x_size,sample_y_size)
    inner_2d = zeros(sample_x_size,sample_y_size)
    for i in 1:sample_x_size
        for j in 1:sample_y_size
            for k in 1:nodes_num
                for t in 1:2
                inner_2d[i,j] += input_weights[k,t]*data[i,j][t]
                end
                inner_2d[i,j] = inner_2d[i,j].+input_biases[k]
                output_2d[i,j] += output_weights[k]*σ(inner_2d[i,j]) 
            end
        end
    end
    return  output_2d.+output_bias
end


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



 function buildLap2D(x,y)
    nx = length(x)
    ny = length(y)

    D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
    Id = I(nx)
    A = kron(D, Id) + kron(Id, D)
    A /= (nx+1)^2
    return A
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



function grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
    grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)= ForwardDiff.jacobian(data->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data, σ),data)
    size_m = size(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ))[1]
    grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)[:,1:size_m])
    grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)[:,size_m+1:end])
    return grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
end

function hess_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
    size_m = size(data)[1]
    #grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)=grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)
    #grad_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)= ForwardDiff.jacobian(data->one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,data, σ),data)
    #grad_2d(input_weights,input_biases,output_weights,output_bias,data, σ) = vcat(grad_x_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ),grad_y_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ))
    hess_x_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(ForwardDiff.jacobian(data->grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)[1],data))
    hess_y_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) = diag(ForwardDiff.jacobian(data->grad_nn_2d(input_weights,input_biases,output_weights,output_bias,data, σ)[2],data)[:,size_m+1:end])
    
    #@show hess_one_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)   
    return hess_x_hidden(input_weights,input_biases,output_weights,output_bias,data, σ) + hess_y_hidden(input_weights,input_biases,output_weights,output_bias,data, σ)
end


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

function obj_2d(input_weights,input_biases,output_weights,output_bias,data,σ)
    g_1 = possion_2d(cos_v_2d(20))[2] #the right-hand side could choose other PDEs
    g_2 = possion_2d(cos_v_2d(20))[3]
    loss = 0
    penalty = 0
    data_size = size(data)[1]
    #separate the data into interior and boundary points
    interior_data, boundary_data=separate_ib(data)
    N_D = size(interior_data)[1]
    N_BC = size(boundary_data)[1]
    for k in 1:N_D
        loss += norm(g_1(interior_data[k,1],interior_data[k,2])+hess_nn_2d(input_weights,input_biases,output_weights,output_bias,interior_data,σ)[k],2)^2
    end
    λ_p = 0.1*sqrt(data_size)
    for l in 1:N_BC
        penalty += norm(g_2(boundary_data[l,1],boundary_data[l,2])-one_hidden_layer_nn(input_weights,input_biases,output_weights,output_bias,boundary_data,σ)[l],2)^2
    end
    loss = loss/(2*N_D)
    #@show whole
    penalty = λ_p*(penalty)/(2*N_BC)
    return penalty
end

#inter = separate_ib(data_data)[1]
#inter = reshape(collect(Iterators.flatten(inter)),length(inter[1]),length(inter))'
#data_data
#hess_nn_2d(ww_try,bb_try,vv_try,dd_try,inter,sigmoid)
#obj_2d(ww_try,bb_try,vv_try,dd_try,data_data,sigmoid)

