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
function LM_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-6
    ϵ = 1e-2
    s_size = 3*size(output_weights)[1]+1
    para_size = size(output_weights)[1]
    #give an initial step sk
    s = 0.001*ones(s_size)
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        mk(s) = mk_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s,λ)
        s =  Optim.minimizer(optimize(mk, s))
        fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
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
    return input_weights,input_biases,output_weights,output_bias
end