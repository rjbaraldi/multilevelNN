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
    ϵ_norm = 0.05
    ϵ_H = 0.1
    κ_H = 0.1
    ϵ_AMG = 1
    A_AMG = matrix_A_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
    fA = sparse(A_AMG)
    fT = sparse(strong_connection(fA,ϵ_AMG)[2])
    fS = sparse(strong_connection(fA,ϵ_AMG)[1])
    splitting = AlgebraicMultigrid.RS_CF_splitting(fS,fT)
    fP = AlgebraicMultigrid.direct_interpolation(fA,fT,splitting)[1]
    fR = AlgebraicMultigrid.direct_interpolation(fA,fT,splitting)[2]
    σ_R = sqrt(det(fR*fR'))
    fR = fR/σ_R
    R_block = vcat(hcat(fR,zeros(size(fR)[1],size(fR)[2]),zeros(size(fR)[1],size(fR)[2]),zeros(size(fR)[1],1)),hcat(zeros(size(fR)[1],size(fR)[2]),fR,zeros(size(fR)[1],size(fR)[2]),zeros(size(fR)[1],1)),hcat(zeros(size(fR)[1],size(fR)[2]),zeros(size(fR)[1],size(fR)[2]),fR,zeros(size(fR)[1],1)),zeros(1,3*size(fR)[2]+1))
    P_block = vcat(hcat(fP,zeros(size(fP)[1],size(fP)[2]),zeros(size(fP)[1],size(fP)[2]),zeros(size(fP)[1],1)),hcat(zeros(size(fP)[1],size(fP)[2]),fP,zeros(size(fP)[1],size(fP)[2]),zeros(size(fP)[1],1)),hcat(zeros(size(fP)[1],size(fP)[2]),zeros(size(fP)[1],size(fP)[2]),fP,zeros(size(fP)[1],1)),zeros(1,3*size(fP)[2]+1))
    R_block[end,end] = 1
    P_block[end,end] = 1
    H_size = size(fP)[2]
    para_size = size(input_biases)[1]
    s_size = 3*para_size+1
    #s = 0.001*ones(s_size)
    s_H = 0.001*ones(3*H_size+1)
    
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ_norm
        #w = copy(input_weights)
        #b = copy(input_biases)
        #v = copy(output_weights)
        #d = copy(output_bias)
        @show norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        g_w = obj_1d_w(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_b = obj_1d_b(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_v = obj_1d_v(input_weights,input_biases,output_weights,output_bias,data,σ)
        g_d = obj_1d_d(input_weights,input_biases,output_weights,output_bias,data,σ)
        grad_obj = vcat(g_w,g_b,g_v,g_d)
        R_grad = R_block*grad_obj
        #if l>1 && norm(R_grad,2) >= κ_H*norm(grad_obj,2) && norm(R_grad,2) > ϵ_H
        #@show norm(R_grad,2) κ_H*norm(grad_obj,2) norm(R_grad,2)
            w_H = fR*input_weights
            b_H = fR*input_biases
            v_H = fR*output_weights
            d_H = output_bias
            #R_p_withoutd = vcat(w_H,b_H,v_H)
            f_H(wH,bH,vH,dH,sH,data,σ) = obj_1d(wH.+sH[1:H_size],bH.+sH[H_size+1:2*H_size],vH.+sH[2*H_size+1:end-1],dH.+sH[end],data,σ)
            grad_H_w(wH,bH,vH,dH,sH,data,σ) = ForwardDiff.gradient(wH -> f_H(wH,bH,vH,dH,sH,data,σ), wH)
            grad_H_b(wH,bH,vH,dH,sH,data,σ) = ForwardDiff.gradient(bH -> f_H(wH,bH,vH,dH,sH,data,σ), bH)
            grad_H_v(wH,bH,vH,dH,sH,data,σ) = ForwardDiff.gradient(vH -> f_H(wH,bH,vH,dH,sH,data,σ), vH)
            grad_H_d(wH,bH,vH,dH,sH,data,σ) = ForwardDiff.gradient(dH -> f_H(wH,bH,vH,dH,sH,data,σ), dH)
            grad_H(wH,bH,vH,dH,sH,data,σ) = vcat(grad_H_w(wH,bH,vH,dH,sH,data,σ),grad_H_b(wH,bH,vH,dH,sH,data,σ),grad_H_v(wH,bH,vH,dH,sH,data,σ),grad_H_d(wH,bH,vH,dH,sH,data,σ))
            mH(s_H) = f_H(w_H,b_H,v_H,d_H,s_H,data,σ)+((R_grad-grad_H(w_H,b_H,v_H,d_H,zeros(3*H_size+1),data,σ))'*s_H)[1] +(λ*norm(s_H,2)^2)/2
            s_H = Optim.minimizer(optimize(mH,s_H))
            s = P_block*s_H
            fk(ss) = obj_1d(input_weights.+ss[1:para_size],input_biases.+ss[para_size+1:2*para_size],output_weights.+ss[2*para_size+1:3*para_size],output_bias.+ss[end],data,σ)
            ρkn = fk(zeros(s_size))-fk(s)
            ρkd = (mH(zeros(3*H_size+1))-mH(s_H))/σ_R
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
        #else
        #   input_weights, input_biases,output_weights,output_bias = LM_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
        #end
    #@show input_weights,input_biases,output_weights,output_bias   
    end
    return input_weights,input_biases,output_weights,output_bias
end
