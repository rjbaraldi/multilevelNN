#In this file, we use FD method rather than AD
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
#FD matrix for laplacian
function buildLap1D(x)
    nx = length(x)
    D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
    D /= 1/(nx-1)^2
    return D
end

#FD matrix for gradient/jacobian
function buildGradient(func,x,δ=1e-6)
    f =func
    nx = length(x)
    output = zeros(nx)
    for i in 1:nx
        ei = zeros(nx)
        ei[i] = 1
        f1 = f(x+δ*ei)[i]
        f2 = f(x-δ*ei)[i]
        output[i] = (f1-f2)/(2*δ)
        output = reshape(output,(nx,1))
    end
    return output
end

function buildJacobian(func,x,δ=1e-6)
    f = func
    nrow = length(f(x))
    ncol = length(x)
    output = zeros(nrow,ncol)
    for i in 1:nrow
        for j in 1:ncol
            ej = zeros(ncol)
            ej[j]=1
            dij = (f(x+δ*ej)[i]-f(x-δ*ej)[i])/(2*δ)
            output[i,j]=dij
        end
    end
    return output
end

function buildGrad1D(x)
    nx= length(x)
    D = diagm(-1=> -ones(nx-1)) + diagm(1=>ones(nx-1))
    D /= 1/(2*(nx-1))
    return D
end




function NN(p,x,r,σ)
    w = reshape(p[1:r],(r,1))
    b = p[r+1:2*r]
    v = reshape(p[2*r+1:3*r],(r,1))
    d = [p[end]]
    return (transpose(v) *σ.(w * transpose(x) .+ b) .+d)'
end

#function second_derivative(σ)
#    σ_1(x) = ForwardDiff.derivative(σ,x)
#   σ_2(x) = ForwardDiff.derivative(σ_1,x)
#    return σ_2
#end

#function nn_lap1D_x(p,x,r,σ)
#    w = reshape(p[1:r],(r,1))
#    b = p[r+1:2*r]
#    v = reshape(p[2*r+1:3*r],(r,1))
#    inner =   w * transpose(x)  .+ b
#    active = second_derivative(σ).(inner)
#    outweight = v.*w.^2
#    output = transpose(outweight) * active
#    return output'
#end 

function NN_xx(p,x,r,σ)
    w = reshape(p[1:r],(r,1))
    b = p[r+1:2*r]
    v = reshape(p[2*r+1:3*r],(r,1))
    nx = length(x)
    x_new = zeros(nx+2)
    x_new[2:end-1].=x
    h = (x[end]-x[1])/(nx-1)
    x_new[1] = x[1]-h
    x_new[end]=x[end]+h
    NN_withoutd(xx) = transpose(v)*σ.(w * transpose(xx) .+ b)
    return (buildLap1D(x_new)*NN_withoutd(x_new)')[2:end-1]
end

function possion_1d(real_solution::Function)
    #@parameters x
    #@variables u(..)
    u(x) = real_solution(x)
    u_grad(x) = ForwardDiff.derivative(x -> u(x),x)
    g_1(x) = -ForwardDiff.derivative(x -> u_grad(x),x)
    g_2(x) = real_solution(x)
    return u,g_1,g_2
end

function cos_v(ν)
    function cos_vx(x)
        return cos(ν*x)
    end
    return cos_vx
end

function constant(a)
    function a_func(x)
        return (-a/2)*x^2
    end
    return a_func
end


function obj(p,x,r,σ)
    g_1 = possion_1d(cos_v(20))[2] #the right-hand side could choose other PDEs
    g_2 = possion_1d(cos_v(20))[3]
    N_u = size(x)[1]-2
    g_1_data = g_1.(x[2:end-1])
    loss = norm(g_1_data+NN_xx(p,x[2:end-1],r,σ),2)^2
    loss = loss/(2*N_u)
    λ_p = 0.1
    penalty = norm(g_2(x[1])-NN(p,x,r,σ)[1],2)^2+norm(g_2(x[end])-NN(p,x,r,σ)[end],2)^2
    penalty = λ_p*penalty/4
    return loss+penalty
end

grad_obj_accurate(p,x,r,σ) = ForwardDiff.gradient(p -> obj(p,x,r,σ),p)

#Test##
x = collect(LinRange(0,1,41))
yreal = cos.(20*x)


#r = 100 with σ=sigmoid, x=LinRange(0,1,41)
using CSV, DataFrames
rw_lu_100 = reshape(CSV.read("rw_lu_100.csv",DataFrame)[!,1],(100,1))
w_lu_100 = CSV.read("rw_lu_100.csv",DataFrame)[!,1]
rb_lu_100 = CSV.read("rb_lu_100.csv",DataFrame)[!,1]
rv_lu_100 = reshape(CSV.read("rv_lu_100.csv",DataFrame)[!,1],(100,1))
v_lu_100 = CSV.read("rv_lu_100.csv",DataFrame)[!,1]
rd_lu_100 = CSV.read("rd_lu_100.csv",DataFrame)[!,1]
obj_1d(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)

rp_lu_100 = vcat(w_lu_100,rb_lu_100,v_lu_100,rd_lu_100)
obj(rp_lu_100,x,100,sigmoid)
J_1(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
J1_accurate(rp_lu_100,x,100,sigmoid)
norm(J_1(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)-J1_accurate(rp_lu_100,x,100,sigmoid))
J_2(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)
J2_accurate(rp_lu_100,x,100,sigmoid)
norm(J_2(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid)-J2_accurate(rp_lu_100,x,100,sigmoid))
s = ones(301)
taylor_ac(rp_lu_100,x,100,sigmoid,s,0.05)
mk_1d(rw_lu_100,rb_lu_100,rv_lu_100,rd_lu_100,x,sigmoid,s,0.05)
function F1(p,x,r,σ)
    g_1 = possion_1d(cos_v(20))[2]
    w = reshape(p[1:r],(r,1))
    b = p[r+1:2*r]
    v = reshape(p[2*r+1:3*r],(r,1))
    nx = length(x)
    x_new = zeros(nx+2)
    x_new[2:end-1].=x
    h = (x[end]-x[1])/(nx-1)
    x_new[1] = x[1]-h
    x_new[end]=x[end]+h
    NN_withoutd(xx) = transpose(v)*σ.(w * transpose(xx) .+ b)
    NN_xx = (buildLap1D(x_new)*NN_withoutd(x_new)')[2:end-1]
    if length(x) == 1
        return g_1(x).+ NN_xx
    else
        return g_1.(x)+NN_xx
    end
end


function F2(p,x,r,σ)
    g_2 = possion_1d(cos_v(20))[3]
    if length(x) == 1
        return g_2(x).-NN(p,x,r,σ)
    else
        return g_2.(x)-NN(p,x,r,σ)
    end
end

function buildJacobian(func,x,δ=1e-6)
    f = func
    nrow = length(f(x))
    ncol = length(x)
    output = zeros(nrow,ncol)
    for i in 1:nrow
        for j in 1:ncol
            ej = zeros(ncol)
            ej[j]=1
            dij = (f(x+δ*ej)[i]-f(x-δ*ej)[i])/(2*δ)
            output[i,j]=dij
        end
    end
    return output
end

function J1_accurate(p,x,r,σ)
    J1_ac(p,x,r,σ) = ForwardDiff.jacobian(p->F1(p,x,r,σ),p)
    return J1_ac(p,x,r,σ)
end

function J1(p,x,r,σ)
    f1(pp) = F1(pp,x,r,σ) 
    Jacobian1 = buildJacobian(f1,p,1e-6)
    return Jacobian1
end



function J2_accurate(p,x,r,σ)
    J2_ac(p,x,r,σ) = ForwardDiff.jacobian(p->F2(p,x,r,σ),p)
    return J2_ac(p,x,r,σ)
end

function J2(p,x,r,σ)
    f2(pp) = F2(pp,x,r,σ)
    Jacobian2 = buildJacobian(f2,p,1e-6)
    return Jacobian2
end

function taylor_ac(p,x,r,σ,s,λ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    f1 = F1(p,x[2:end-1],r,σ)
    j1 = J1_accurate(p,x[2:end-1],r,σ)
    first_term = norm(f1,2)^2+(2*(j1'*f1)'*s)[1]+s'*j1'*j1*s
    f20 = F2(p,x[1],r,σ)
    f2e = F2(p,x[end],r,σ)
    j20 = J2_accurate(p,x[1],r,σ)
    j2e = J2_accurate(p,x[end],r,σ)
    second_term = norm(f20,2)^2+(2*(j20'*f20)'*s)[1]+s'*j20'*j20*s+norm(f2e,2)^2+(2*(j2e'*f2e)'*s)[1]+s'*j2e'*j2e*s
    return first_term/(2*N_D)+λ_p*second_term/(2*N_BC)+λ*norm(s,2)^2
end


function taylork(p,x,r,σ,s,λ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    f1 = F1(p,x[2:end-1],r,σ)
    j1 = J1(p,x[2:end-1],r,σ)
    first_term = norm(f1,2)^2+(2*(j1'*f1)'*s)[1]+s'*j1'*j1*s
    f20 = F2(p,x,r,σ)[1]
    f2e = F2(p,x,r,σ)[end]
    j20 = J2(p,x,r,σ)[1,:]
    j2e = J2(p,x,r,σ)[end,:]
    second_term = norm(f20,2)^2+(2*(j20'*f20)'*s)[1]+s'*j20'*j20*s+norm(f2e,2)^2+(2*(j2e'*f2e)'*s)[1]+s'*j2e'*j2e*s
    return first_term/(2*N_D)+λ_p*second_term/(2*N_BC)+λ*norm(s,2)^2
end

function mk_A_ac(p,x,r,σ,λ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    s_size = 3*r+1
    j1 = J1_accurate(p,x[2:end-1,:],r,σ)
    first_term = j1'*j1
    j20 = J2_accurate(p,x[1],r,σ)
    j2e = J2_accurate(p,x[end],r,σ)
    second_term = j20'*j20+j2e'*j2e
    return first_term/(N_D)+λ_p*second_term/N_BC+λ*Matrix{Float64}(I, s_size, s_size) 
end



function mk_A(p,x,r,σ,λ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    s_size = 3*r+1
    j1 = J1(p,x[2:end-1],r,σ)
    first_term = j1'*j1
    j20 = J2(p,x[1],r,σ)
    j2e = J2(p,x[end],r,σ)
    second_term = j20'*j20+j2e'*j2e
    return first_term/(N_D)+λ_p*second_term/N_BC+λ*Matrix{Float64}(I, s_size, s_size) 
end

function mk_b_ac(p,x,r,σ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    s_size = 3*r+1
    first_term = zeros(s_size)
    f1 = F1(p,x[2:end-1],r,σ)
    j1 = J1_accurate(p,x[2:end-1],r,σ)
    first_term = j1'*f1
    f20 = F2(p,x[1],r,σ)
    f2e = F2(p,x[end],r,σ)
    j20 = J2_accurate(p,x[1],r,σ)
    j2e = J2_accurate(p,x[end],r,σ)
    second_term = j20'*f20+j2e'*f2e
    return first_term/(N_D)+λ_p*second_term/(N_BC)
end

function mk_b(p,x,r,σ)
    N_D = length(x)-2
    N_BC = 2
    λ_p = 0.1
    s_size = 3*r+1
    first_term = zeros(s_size)
    f1 = F1(p,x[2:end-1],r,σ)
    j1 = J1(p,x[2:end-1],r,σ)
    first_term = j1'*f1
    f20 = F2(p,x[1],r,σ)
    f2e = F2(p,x[end],r,σ)
    j20 = J2(p,x[1],r,σ)
    j2e = J2(p,x[end],r,σ)
    second_term = j20'*f20+j2e'*f2e
    return first_term/(N_D)+λ_p*second_term/(N_BC)
end


function CGLM_AP(p,x,r,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-4
    s_size = 3*r+1
    #give an initial step sk
    s = 0.001*ones(s_size)
    iteration_all = 0
    iteration = 0
    m_k(s) = taylor_ac(p,x,r,σ,s,λ)
    fk(s) = obj(p.+s,x,r,σ)
    while norm(grad_obj_accurate(p,x,r,σ),2) >= ϵ && m_k(zeros(s_size))-m_k(s) != 0
        @show obj(p,x,r,σ) norm(grad_obj_accurate(p,x,r,σ),2)
        s = cg(mk_A_ac(p,x,r,σ,λ),(-1).*vec(mk_b_ac(p,x,r,σ)))[1]
        iteration_all += cg(mk_A_ac(p,x,r,σ,λ),(-1).*vec(mk_b_ac(p,x,r,σ)))[2].niter
        #@show s
        ρkn = fk(zeros(s_size))-fk(s)
        ρkd = m_k(zeros(s_size))-m_k(s)
        ρ = ρkn/ρkd
        @show ρkn ρkd ρ
        if ρ >= η1
            p = p .+ s
            if ρ >= η2
                λ = max(λ_min,γ2*λ)
            else
                λ = max(λ_min,γ1*λ)
            end
        else
            p = p
            λ = γ3*λ
        end
        iteration += 1
    end
    println("Iteration: $iteration", "Total Iteration: $iteration_all")
    return p
end



function CGLSLM_AP(p,x,r,σ)
    η1 = 0.1
    η2 = 0.75
    γ1 = 0.85
    γ2 = 0.5
    γ3 = 1.5
    λ = 0.05
    λ_min = 1e-4
    ϵ = 1e-4
    s_size = 3*r+1
    #give an initial step sk
    s = 0.001*ones(s_size)
    iteration_all = 0
    iteration = 0
    m_k(s) = taylor_ac(p,x,r,σ,s,λ)
    fk(s) = obj(p.+s,x,r,σ)
    while norm(grad_obj_accurate(p,x,r,σ),2) >= ϵ && m_k(zeros(s_size))-m_k(s) != 0
        @show obj(p,x,r,σ) norm(grad_obj_accurate(p,x,r,σ),2)
        s = cgls(mk_A_ac(p,x,r,σ,λ),(-1).*vec(mk_b_ac(p,x,r,σ)))[1]
        iteration_all += cgls(mk_A_ac(p,x,r,σ,λ),(-1).*vec(mk_b_ac(p,x,r,σ)))[2].niter
        #@show s
        ρkn = fk(zeros(s_size))-fk(s)
        ρkd = m_k(zeros(s_size))-m_k(s)
        ρ = ρkn/ρkd
        @show ρkn ρkd ρ
        if ρ >= η1
            p = p .+ s
            if ρ >= η2
                λ = max(λ_min,γ2*λ)
            else
                λ = max(λ_min,γ1*λ)
            end
        else
            p = p
            λ = γ3*λ
        end
        iteration += 1
    end
    println("Iteration: $iteration", "Total Iteration: $iteration_all")
    return p
end



###Test###
#the real solution is u(x)=cos(20x)
#r=100, x=LinRange(0,1,41)
#load initial data
using CSV, DataFrames
w_lu_100 = CSV.read("rw_lu_100.csv",DataFrame)[!,1]
b_lu_100 = CSV.read("rb_lu_100.csv",DataFrame)[!,1]
v_lu_100 = CSV.read("rv_lu_100.csv",DataFrame)[!,1]
d_lu_100 = CSV.read("rd_lu_100.csv",DataFrame)[!,1]
p_lu_100 = vcat(w_lu_100,b_lu_100,v_lu_100,d_lu_100)
x=collect(LinRange(0,1,41))
y_real = cos.(20*x)
#CG
@time begin
p_cos20_100cg = CGLM_AP(p_lu_100,x,100,sigmoid)
end

y_pred_cos20_100cg = NN(p_cos20_100cg,x,100,sigmoid)
norm(y_real-y_pred_cos20_100cg)/41
#write data into files and plot 
p1cg_cos20 = DataFrame([p_cos20_100cg],:auto)
CSV.write("p_cos20_100cg.csv",p1cg_cos20)
y_cos20_100cg = DataFrame(y_pred_cos20_100cg,:auto)
CSV.write("y_pred_cos20_100cg.csv",y_cos20_100cg)
using Plots
Plots.plot(x,y_pred_cos20_100cg,label="NN_CG100",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)

#CGLS
@time begin
    p_cos20_100cgls = CGLSLM_AP(p_lu_100,x,100,sigmoid)
end
    
y_pred_cos20_100cgls = NN(p_cos20_100cgls,x,100,sigmoid)
norm(y_real-y_pred_cos20_100cgls)/41
#write data into files and plot 
p1cgls_cos20 = DataFrame([p_cos20_100cgls],:auto)
CSV.write("p_cos20_100cgls.csv",p1cgls_cos20)
y_cos20_100cgls = DataFrame(y_pred_cos20_100cgls,:auto)
CSV.write("y_pred_cos20_100cgls.csv",y_cos20_100cgls)
using Plots
Plots.plot(x,y_pred_cos20_100cgls,label="NN_CGLS100",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)



#####r=300
#load the initial data

w_lu_300 = CSV.read("rw_lu_300.csv",DataFrame)[!,1]
b_lu_300 = CSV.read("rb_lu_300.csv",DataFrame)[!,1]
v_lu_300 = CSV.read("rv_lu_300.csv",DataFrame)[!,1]
d_lu_300 = CSV.read("rd_lu_300.csv",DataFrame)[!,1]
p_lu_300 = vcat(w_lu_300,b_lu_300,v_lu_300,d_lu_300)


#CG
@time begin
    p_cos20_300cg = CGLM_AP(p_lu_300,x,300,sigmoid)
end
y_pred_cos20_300cg = NN(p_cos20_300cg,x,300,sigmoid)
norm(y_real-y_pred_cos20_300cg)/41
#write data into files and plot 
p3cg_cos20 = DataFrame([p_cos20_300cg],:auto)
CSV.write("p_cos20_300cg.csv",p3cg_cos20)
y_cos20_300cg = DataFrame(y_pred_cos20_300cg,:auto)
CSV.write("y_pred_cos20_300cg.csv",y_cos20_300cg)
using Plots
Plots.plot(x,y_pred_cos20_300cg,label="NN_CG300",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)


#CGLS
@time begin
    p_cos20_300cgls = CGLSLM_AP(p_lu_300,x,300,sigmoid)
end
    
y_pred_cos20_300cgls = NN(p_cos20_300cgls,x,300,sigmoid)
norm(y_real-y_pred_cos20_300cgls)/41
#write data into files and plot 
p3cgls_cos20 = DataFrame([p_cos20_300cgls],:auto)
CSV.write("p_cos20_300cgls.csv",p3cgls_cos20)
y_cos20_300cgls = DataFrame(y_pred_cos20_300cgls,:auto)
CSV.write("y_pred_cos20_300cgls.csv",y_cos20_300cgls)
using Plots
Plots.plot(x,y_pred_cos20_300cgls,label="NN_CGLS300",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)

#####r=500
#load the initial data

w_lu_500 = CSV.read("rw_lu_500.csv",DataFrame)[!,1]
b_lu_500 = CSV.read("rb_lu_500.csv",DataFrame)[!,1]
v_lu_500 = CSV.read("rv_lu_500.csv",DataFrame)[!,1]
d_lu_500 = CSV.read("rd_lu_500.csv",DataFrame)[!,1]
p_lu_500 = vcat(w_lu_500,b_lu_500,v_lu_500,d_lu_500)

#CG
@time begin
    p_cos20_500cg = CGLM_AP(p_lu_500,x,500,sigmoid)
end
y_pred_cos20_500cg = NN(p_cos20_500cg,x,500,sigmoid)
norm(y_real-y_pred_cos20_500cg)/41
#write data into files and plot 
p5cg_cos20 = DataFrame([p_cos20_500cg],:auto)
CSV.write("p_cos20_500cg.csv",p5cg_cos20)
y_cos20_500cg = DataFrame(y_pred_cos20_500cg,:auto)
CSV.write("y_pred_cos20_500cg.csv",y_cos20_500cg)
using Plots
Plots.plot(x,y_pred_cos20_500cg,label="NN_CG500",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)

#CGLS

@time begin
    p_cos20_500cgls = CGLSLM_AP(p_lu_500,x,300,sigmoid)
end
    
y_pred_cos20_500cgls = NN(p_cos20_500cgls,x,500,sigmoid)
norm(y_real-y_pred_cos20_500cgls)/41
#write data into files and plot 
p5cgls_cos20 = DataFrame([p_cos20_500cgls],:auto)
CSV.write("p_cos20_500cgls.csv",p5cgls_cos20)
y_cos20_500cgls = DataFrame(y_pred_cos20_500cgls,:auto)
CSV.write("y_pred_cos20_500cgls.csv",y_cos20_500cgls)
using Plots
Plots.plot(x,y_pred_cos20_500cgls,label="NN_CGLS500",seriestype=:scatter)
show()
Plots.plot!(x,y_real,label="real_solution",linewidth=2.0)