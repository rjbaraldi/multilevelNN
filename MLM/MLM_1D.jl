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

function mH_1d(wh,bh,vh,dh,data,σ,sH,λ,R)
   wH = R*wh
   bH = R*bh
   vH = R*vh
   dH = dh
   R_block = vcat(hcat(R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],1)),zeros(1,3*size(R)[2]+1))
   R_block[end,end] = 1
   first_term = mk_1d(wH,bH,vH,dH,data,σ,sH,λ)
   second_term = ((R_block*grad_obj_1d(wh,bh,vh,dh,data,σ)-grad_obj_1d(wH,bH,vH,dH,data,σ))'*sH)[1]
   return first_term+second_term
end



function mH_line_A(wh,bh,vh,dh,data,σ,sH,λ,R)
   wH = R*wh
   bH = R*bh
   vH = R*vh
   dH = dh
   sH_size = size(sH)[1] 
   return line_mk_A(wH,bH,vH,dH,data,σ,sH_size,λ)
end

function mH_line_b(wh,bh,vh,dh,data,σ,sH,λ,R)
   wH = R*wh
   bH = R*bh
   vH = R*vh
   dH = dh
   R_block = vcat(hcat(R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],1)),hcat(zeros(size(R)[1],size(R)[2]),zeros(size(R)[1],size(R)[2]),R,zeros(size(R)[1],1)),zeros(1,3*size(R)[2]+1))
   R_block[end,end] = 1
   sH_size = size(sH)[1] 
   return line_mk_b(wH,bH,vH,dH,data,σ,sH_size,λ)+R_block*grad_obj_1d(wh,bh,vh,dh,data,σ)-grad_obj_1d(wH,bH,vH,dH,data,σ)
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
    ϵ = 1e-6
    ϵ_H = ϵ
    κ_H = 0.1
    ϵ_AMG = 0.9
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
    s = 0.001*ones(s_size)
    s_H = 0.001*ones(3*H_size+1)
    while norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2) > ϵ
        @show obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ) norm(grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ),2)
        grad_obj = grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        R_grad = R_block*grad_obj_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
        if l>1 && norm(R_grad,2) >= κ_H*norm(grad_obj,2) && norm(R_grad,2) > ϵ_H
            mH(s_H) = mH_1d(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR)
            s_H = vec(cholesky(mH_line_A(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR))\((-1)*mH_line_b(input_weights,input_biases,output_weights,output_bias,data,σ,s_H,λ,fR)))
            s = P_block*s_H
            mh(s_H) = mH(s_H)/σ_R
            fk(s) = obj_1d(input_weights.+s[1:para_size],input_biases.+s[para_size+1:2*para_size],output_weights.+s[2*para_size+1:3*para_size],output_bias.+s[end],data,σ)
            @show fk(s) fk(zeros(s_size))
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
            input_weights,input_biases,output_weights,output_bias = LM_1D(input_weights,input_biases,output_weights,output_bias,data,σ)
        end
    end
    return input_weights,input_biases,output_weights,output_bias
end


lower_bound = -1
upper_bound = 3
rw_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rb_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500)
rv_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(500,1)
rd_lu_500 = lower_bound.+(upper_bound-lower_bound).*rand(1)
#ϵ_AMG = 0.9
#    A_AMG = matrix_A_1D(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid)
#    fA = sparse(A_AMG)
#    fT = sparse(strong_connection(fA,ϵ_AMG)[2])
#    fS = sparse(strong_connection(fA,ϵ_AMG)[1])
#    splitting = AlgebraicMultigrid.RS_CF_splitting(fS,fT)
#    fP = AlgebraicMultigrid.direct_interpolation(fA,fT,splitting)[1]
#    fR = AlgebraicMultigrid.direct_interpolation(fA,fT,splitting)[2]
@time begin
    mrw_lu_500,mrb_lu_500,mrv_lu_500,mrd_lu_500 = MLM_1D(rw_lu_500,rb_lu_500,rv_lu_500,rd_lu_500,x,sigmoid,2)
    mre_lu_500 = obj_1d(mrw_lu_500,mrb_lu_500,mrv_lu_500,mrd_lu_500,x,sigmoid)
    mrad_lu_500 = norm(grad_obj_1d(mrw_lu_500,mrb_lu_500,mrv_lu_500,mrd_lu_500,x,sigmoid),2)
end
