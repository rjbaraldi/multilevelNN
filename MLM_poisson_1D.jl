using LeastSquaresOptim
using TrustRegionMethods
using PyPlot
using DifferentialEquations
using Optim, Flux
using Random
using LinearAlgebra
using AlgebraicMultigrid
using ForwardDiff, SparseArrays

#Define the pde
function pde_sol(x,v)
    l=length(x)
    u=zeros(Float32,l,1)
    g1=u
    g2=u
    for k=1:l
        u[k] = cos(v*x[k])
        g1[k] = v^2*cos(v*x[k])
    end
    g2[1]=cos(v*x[1])
    g2[l]=cos(v*x[l])
    return u,g1,g2
end



#Define laplace u_T
function buildLap1D(x)
    nx = length(x)
    D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
    D /= (nx-1)^2
    return D
end


#Define neural network
function one_nn(ww,xx,bb,dd,vv)
    NN_out=0
    r = length(ww)
    t = length(xx)
    for k = 1:r
        for l = 1:t
            NN_out += vv[k]*sigmoid(ww[k]*xx[l]+bb[k])
        end
    end
    return NN_out+dd
end
Dx(ww,xx,bb,dd,vv) = ForwardDiff.gradient(xx -> one_nn(ww,xx,bb,dd,vv),xx)
Dxx(ww,xx,bb,dd,vv) = ForwardDiff.jacobian(xx -> Dx(ww,xx,bb,dd,vv),xx)


#Define the objective function
function obj(ww,xx,bb,dd,vv)
    nx = length(xx)
    u,g1,g2 = pde_sol(xx,20)
    DG1 = sum((g1 .- diag(Dxx(ww,xx,bb,dd,vv))).^2)
    DG2 = (g2[1]-one_nn(ww,xx[1],bb,dd,vv))^2+(g2[nx]-one_nn(ww,xx[nx],bb,dd,vv))^2
    return (1/2*nx)*DG1+(1/4)*DG2
end
#Find the Jacobian matrix
obj_v(ww,xx,bb,dd,vv) = ForwardDiff.gradient(vv -> obj(ww,xx,bb,dd,vv),vv)
obj_w(ww,xx,bb,dd,vv) = ForwardDiff.gradient(ww -> obj(ww,xx,bb,dd,vv),ww)
obj_b(ww,xx,bb,dd,vv) = ForwardDiff.gradient(bb -> obj(ww,xx,bb,dd,vv),bb)

#Define the matrix A
function matrix_A(ww,xx,bb,dd,vv)
    return obj_v(ww,xx,bb,dd,vv)*transpose(obj_v(ww,xx,bb,dd,vv))/norm(obj_v(ww,xx,bb,dd,vv),Inf)+obj_w(ww,xx,bb,dd,vv)*transpose(obj_w(ww,xx,bb,dd,vv))/norm(obj_w(ww,xx,bb,dd,vv),Inf)+obj_b(ww,xx,bb,dd,vv)*transpose(obj_b(ww,xx,bb,dd,vv))/norm(obj_b(ww,xx,bb,dd,vv),Inf)
end
### some example data to test the codes
#v = 20
#x_data = LinRange(0.0,1.0,2*v+1)
#xx_try=collect(x_data)
#ww_try=[1.0,2.0,3.0]
#bb_try=[0.0,1.0,2.0]
#vv_try=[2.0,1.0,1.0]
#dd_try=10.0




#Define the prolongation and restriction operator
#Step 1 Define the classical strength of connection
#Principle: |A[i,j]| >= theta * max(|A[i,k]|), where k != i 
#Parameters
#----------
#A: the matrix comes from our problem
#θ: the threshold parameter in [0,1]
#----------
#Returns
#T: Matrix graph defining strong connections. T[i,j]=1 if vertex i is strongly influenced by vertex j; otherwise, T[i,j]=0
    function find_max_off_diag(A, i)
        m = zero(eltype(A))
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if row != i
                m = max(m, abs(val))
            end
        end
        m
    end
    
    function find_max(A, i)
        m = zero(eltype(A))
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            m = max(m, val)
        end
        m
    end
    
    
    function scale_cols_by_largest_entry!(A)
        n = size(A, 1)
        for i = 1:n
            _m = find_max(A, i)
            for j in nzrange(A, i)
                A.nzval[j] /= _m
            end
        end
        A
    end
    
    function strength_classical(A,θ)
        m,n = size(A)
        T = copy(A)
        for i = 1:n
            _m = find_max_off_diag(T,i)
            threshold = θ*_m
            for j in nzrange(T,i)
                row = T.rowval[j]
                val = T.nzval[j]
                if row != i 
                    if abs(val) >= threshold
                        T.nzval[j] = 1
                    else
                        T.nzval[j] = 0
                    end
                end
            end
        end
        dropzeros!(T)
        scale_cols_by_largest_entry!(T)
        T
    end

#Step 2 Compute a C/F splitting using Ruge-Stuben coarsening
#Parameters
#----------
#S: strength of connection matrix indicating the strength between nodes i and j (S[i,j])
#----------
#Returns
#splitting: Array of length of S of ones (coarse) and zeros (fine) (haven't finished), should base on Figure 2 in our meeting notes
function splitting(S)
    n = size (S,1)
    lambda = zeros(Int,n)
    T = adjoint(S)
    Tp = T.colptr
    Tj = T.rowval
    Sp = S.colptr
    Sj = S.rowval
    interval_ptr = zeros(Int,n+1)
    interval_count = zeros(Int,n+1)
    index_to_node = zeros(Int,n)
    node_to_index = zeros(Int,n)
    for i =1:n
        #compute lambda[i], which is the number of nodes strongly coupled to node i
        lambda[i] = Sp[i+1]-Sp[i]
        interval_count[lambda[i]+1] += 1
    end



















    
    





   




