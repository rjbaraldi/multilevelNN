#In this file, we give the codes for MLM in 1D

function matrix_A_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
    F_1w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_1b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_1v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_1(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    F_2w(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_weights)
    F_2b(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(input_biases->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),input_biases)
    F_2v(input_weights,input_biases,output_weights,output_bias,data,σ) = ForwardDiff.jacobian(output_weights->F_2(input_weights,input_biases,output_weights,output_bias,data,σ),output_weights)
    J1w = zeros(1,size(input_weights)[1])
    J1b = J1w
    J1v = J1b
    for i in 2:length(data)-1
        J1w .+= F_1w(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1b .+= F_1b(input_weights,input_biases,output_weights,output_bias,data[i],σ)
        J1v .+= F_1v(input_weights,input_biases,output_weights,output_bias,data[i],σ)
    end
    J2w = F_2w(input_weights,input_biases,output_weights,output_bias,data[1],σ)+F_2w(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J2b = F_2b(input_weights,input_biases,output_weights,output_bias,data[1],σ)+F_2b(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    J2v = F_2v(input_weights,input_biases,output_weights,output_bias,data[1],σ)+F_2v(input_weights,input_biases,output_weights,output_bias,data[end],σ)
    Jw = J1w/(2*(length(data)-2))+0.1*length(data)*J2w/4
    Jb = J1b/(2*(length(data)-2))+0.1*length(data)*J2b/4
    Jv = J1v/(2*(length(data)-2))+0.1*length(data)*J2v/4
    return Jw'*Jw/norm(Jw,Inf)+Jb'*Jb/norm(Jb,Inf)+Jv'*Jv/norm(Jv,Inf)
end
function MLM_1D(input_weights,input_biases,output_weights,output_bias,data,σ,l)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    #θ = 1e-2
    λ_min = 1e-6
    ϵ = 5e-2
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
    A_AMG = matrix_A_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
    P = prolongation(A_AMG, ϵ_AMG)
    R = transpose(P)
    σ_R = sqrt(det(R*R'))
    R = R/σ_R
    R_block = vcat(hcat(R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],1)),zeros(1,3*size(R)[2]+1))
    P_block = vcat(hcat(P,zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],1)),hcat(zeros(size(P)[1],size(P)[2]),P,zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],1)),hcat(zeros(size(P)[1],size(P)[2]),zeros(size(P)[1],size(P)[2]),P,zeros(size(P)[1],1)),zeros(1,3*size(P)[2]+1))
    R_block[end,end] = 1
    P_block[end,end] = 1
    H_size = size(P)[2]
    para_size = size(input_biases)[1]
    s_size = 3*para_size+1
    s = 0.001*ones(s_size)
    s_H = 0.001*ones(3*H_size+1)
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        g_w = obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_b = obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_v = obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_d = obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj = vcat(g_w,g_b,g_v,g_d)
        R_grad = R_block*grad_obj
        if l>1 && norm(R_grad,2) >= κ_H*norm(grad_obj,2) && norm(R_grad,2) > ϵ_H
            w_H = R*input_weights
            b_H = R*input_biases
            v_H = R*output_weights
            d_H = output_bias
            R_p_withoutd = vcat(w_H,b_H,v_H)
            f_H(w_H,b_H,v_H,d_H,s_H,data,σ) = obj_1d(w_H.+s_H[1:H_size],b_H.+s_H[H_size+1:2*H_size],v_H.+s_H[2*H_size+1:end-1],d_H.+s_H[end],data,σ)
            grad_H_w(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(w_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), w_H)
            grad_H_b(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(b_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), b_H)
            grad_H_v(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(v_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), v_H)
            grad_H_d(w_H,b_H,v_H,d_H,s_H,data,σ) = ForwardDiff.gradient(d_H -> f_H(w_H,b_H,v_H,d_H,s_H,data,σ), d_H)
            grad_H(w_H,b_H,v_H,d_H,s_H,data,σ) = vcat(grad_H_w(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_b(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_v(w_H,b_H,v_H,d_H,s_H,data,σ),grad_H_d(w_H,b_H,v_H,d_H,s_H,data,σ))
            mH(s_H) = f_H(w_H,b_H,v_H,d_H,s_H,data,σ)+((R_grad-grad_H(w_H,b_H,v_H,d_H,zeros(3*H_size+1),data,σ))'*s_H)[1] +(λ*norm(s_H,2)^2)/2
            return nwH = MLM_1D(w_H,b_H,v_H,d_H,data,σ,l-1)[1],nbH = MLM_1D(w_H,b_H,v_H,d_H,data,σ,l-1)[2],nvH = MLM_1D(w_H,b_H,v_H,d_H,data,σ,l-1)[3],ndH = MLM_1D(w_H,b_H,v_H,d_H,data,σ,l-1)[4]
            s_H = vcat(nwH,nbH,nvH,ndH)
            s = P_block*vcat(nwH-w_H,nbH-b_H,nvH-v_H,ndH-d_H)
            mh(s_H) = mH(s_H)/σ_R
            fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = mh(zeros(3*H_size+1))-mh(s_H)
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
        else
            LM_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
        end
    end
    return input_weights,input_biases,output_weights,output_bias
end

