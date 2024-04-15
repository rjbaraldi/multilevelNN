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
        
        s = cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
        @show cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))
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

function LM_1D_CGLS(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-6
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
        @show cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))
     
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
yreal = -x.^2/2
lower_bound = -5
upper_bound = 5
rw_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rb_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500)
rv_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rd_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(1)

@time begin
    lrw_lu_500_d,lrb_lu_500_d,lrv_lu_500_d,lrd_lu_500_d = LM_1D(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
    lrerr_lu_500_d = norm(vec(one_hidden_layer_nn(lrw_lu_100_d,lrb_lu_500_d,lrv_lu_500_d,lrd_lu_500_d,x,sigmoid).-yreal),2)
end
@time begin
    lrw_lu_500_cg,lrb_lu_500_cg,lrv_lu_500_cg,lrd_lu_500_cg = LM_1D_CG(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
    lrerr_lu_500_cg = norm(vec(one_hidden_layer_nn(lrw_lu_500_cg,lrb_lu_500_cg,lrv_lu_500_cg,lrd_lu_500_cg,x,sigmoid).-yreal),2)
end
@time begin
    lrw_lu_500_cgls,lrb_lu_500_cgls,lrv_lu_500_cgls,lrd_lu_500_cgls = LM_1D_CGLS(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
    lrerr_lu_500_cgls = norm(vec(one_hidden_layer_nn(lrw_lu_500_cgls,lrb_lu_500_cgls,lrv_lu_500_cgls,lrd_lu_500_cgls,x,sigmoid).-yreal),2)
end


function grad_check(input_weights,input_biases,output_weights,output_bias,θ,data,σ)
    size_w = size(input_weights)[1]
    size_b = size(input_biases)[1]
    size_v = size(output_weights)[1]
    grad = grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
    gradapprox = zeros(size_w+size_b+size_v+1)
    fw(x) = obj_1d(input_weights.+x,input_biases,output_weights,output_bias,data,σ)
    gw(x) = (fw(x)-fw(-x))/(2*θ)
    fb(x) = obj_1d(input_weights,input_biases.+x,output_weights,output_bias,data,σ)
    gb(x) = (fb(x)-fb(-x))/(2*θ)
    fv(x) = obj_1d(input_weights,input_biases,output_weights.+x,output_bias,data,σ)
    gv(x) = (fv(x)-fv(-x))/(2*θ)
    for i in 1:size_w
    e = zeros(size_w)
    e[i] = θ
    gradapprox[i] = gw(e)
    gradapprox[size_w+i] = gb(e)
    gradapprox[size_w+size_b+i] = gv(e)
    end
    fd(θ) = obj_1d(input_weights,input_biases,output_weights,output_bias.+θ,data,σ)
    gd(θ) = (fd(θ)-fd(-θ))/(2*θ)
    gradapprox[end] = gd(θ)
    @show norm(gradapprox.- grad,2) norm(grad,2) norm(gradapprox,2)
return norm(grad.-gradapprox,2)/(norm(grad,2)+norm(gradapprox))
end


##########TEST#############
grad_check_500_cg = grad_check(lrw_lu_500_cg,lrb_lu_500_cg,lrv_lu_500_cg,lrd_lu_500_cg,1e-4,x,sigmoid)
