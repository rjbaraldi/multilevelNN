#In this file, we give the codes for MLM in 1D


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





function mH_1D(wh,bh,vh,dh,data,σ,sH,λ,R)
    wH = R*wh
    bH = R*bh
    vH = R*vh
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

function mH_line_A(wh,bh,vh,dh,data,σ,sH_size,λ,R)
   wH = R*wh
   bH = R*bh
   vH = R*vh
   dH = dh
   return line_mk_A(wH,bH,vH,dH,data,σ,sH_size,λ)
end

function mH_line_b(wh,bh,vh,dh,data,σ,sH_size,λ,R)
   wH = R*wh
   bH = R*bh
   vH = R*vh
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
    fP = prolongation(A,ϵ_AMG)
    fR = transpose(fP)#/norm(fP)
    #fP = transpose(fR)
    #σ_R = norm(fP)
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
            mH(s_H) = mH_1D(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR)
            s_H = cg(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR)))[1]
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
            s = cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
            @show cg(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2:end]
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
    A_try = matrixA(input_weights,input_biases,output_weights,output_bias,data,σ)
    fP = prolongation(A_try,ϵ_AMG)
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
            mH(s_H) = mH_1D(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR)
            s_H = cgls(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR)))[1]
            @show cgls(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR),(-1).*vec(mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,sH_size,λ,fR)))[2:end]
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
            s = cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[1]
            @show cgls(line_mk_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ),(-1).*vec(line_mk_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_size,λ)))[2:end]
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
##########Test#################
x = collect(LinRange(0,1,2000))
yreal = -x.^2/2
lower_bound = -5
upper_bound = 5
rw_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rb_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500)
rv_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rd_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(1)

@time begin
    mrw_lu_500_cg,mrb_lu_500_cg,mrv_lu_500_cg,mrd_lu_500_cg = MLM_1D_CG(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid,2)
    mrerr_lu_500 = norm(vec(one_hidden_layer_nn(mrw_lu_500_cg,mrb_lu_500_cg,mrv_lu_500_cg,mrd_lu_500_cg,x,sigmoid).-yreal),2)
end
@time begin
    mrw_lu_500_cgls,mrb_lu_500_cgls,mrv_lu_500_cgls,mrd_lu_500_cgls = MLM_1D_CGLS(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid,2)
    mrerr_lu_500_cgls = norm(vec(one_hidden_layer_nn(mrw_lu_500_cgls,mrb_lu_500_cgls,mrv_lu_500_cgls,mrd_lu_500_cgls,x,sigmoid).-yreal),2)
end
