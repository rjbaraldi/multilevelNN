#In this file, we give the codes for LM in 1D case.
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

function line_mk(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
    N_D = length(data)-2
    N_BC = 2
    λ_p = 0.1*length(data)
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
    λ_p = 0.1*length(data)
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
    λ_p = 0.1*length(data)
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
    λ_min = 1e-6
    ϵ = 1e-6
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


############Test#################
x = collect(LinRange(0,1,41))
rw = rand(100,1)
rb = rand(100)
rv = rand(100,1)
rd = rand(1)
@time begin
    lrw,lrb,lrv,lrd = LM_1D(rw,rb,rv,rd,x,sigmoid)
    lre = obj_1d(lrw,lrb,lrv,lrd,x,sigmoid)
end
rw_200 = rand(200,1)
rb_200 = rand(200)
rv_200 = rand(200,1)
rd_200 = rand(1)
@time begin
    lrw_200,lrb_200,lrv_200,lrd_200 = LM_1D(rw_200,rb_200,rv_200,rd_200,x,sigmoid)
    lre_200 = obj_1d(lrw_200,lrb_200,lrv_200,lrd_200,x,sigmoid)
end


lower_bound = -1
upper_bound = 3
rw_lu_100 = lower_bound.+(upper_bound-lower_bound).*rand(100,1)
rb_lu_100 = lower_bound.+(upper_bound-lower_bound).*rand(100)
rv_lu_100 = lower_bound.+(upper_bound-lower_bound).*rand(100,1)
rd_lu_100 = lower_bound.+(upper_bound-lower_bound).*rand(1)
@time begin
    lrw_lu_100,lrb_lu_100,lrv_lu_100,lrd_lu_100 = LM_1D(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
    lre_lu_200 = obj_1d(lrw_lu_100,lrb_lu_100,lrv_lu_100,lrd_lu_100,x,sigmoid)
    grad_lu_200 = norm(grad_obj_1d(lrw_lu_100,lrb_lu_100,lrv_lu_100,lrd_lu_100,x,sigmoid),2)
end
