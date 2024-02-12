using LeastSquaresOptim
using TrustRegionMethods
using PyPlot
using DifferentialEquations
using Optim, Flux
using Random
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
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
    #remove the diagonal elements in A
       A = tril(A,-1)+triu(A,1)
       max_row = maximum(abs.(A)[i,:])
       return max_row
    end
    
    
    function find_max(A, i)
       return maximum(abs.(A)[i,:])
    end
    
    
    function strong_connection(A,θ)
        m,n = size(A)
        T = copy(A)
        for i = 1:n
            _m = find_max_off_diag(T,i)
            threshold = θ*_m
            for j in 1:m
                if abs(T[i,j]) >= threshold
                    T[i,j] = 1
                else
                    T[i,j] = 0
                end
            end
        end
        #T[diagind(T)] .= 1
        return T
    end
function positive_couplings(A,θ)
    m,n = size(A)
    T = copy(A)
    for i = 1:n
        _m = find_max_off_diag(T,i)
        threshold = θ*_m
        for j in 1:m
            if T[i,j] >= threshold
                T[i,j] = 1
            else
                T[i,j] = 0
            end
        end
    end
    return T
end

function negative_couplings(A,θ)
    m,n = size(A)
    T = copy(A)
    for i = 1:n
        _m = find_max_off_diag(T,i)
        threshold = θ*_m
        for j in 1:m
            if -T[i,j] >= threshold
                T[i,j] = 1
            else
                T[i,j] = 0
            end
        end
    end
    return T
end

#Step 2 Compute a C/F splitting using Ruge-Stuben coarsening
#Parameters
#----------
#S: strength of connection matrix indicating the strength between nodes i and j (S[i,j])
#----------
#Returns
#splitting: Array of length of S of ones (coarse) and zeros (fine) 
const F_NODE = 0
const C_NODE = 1
const U_NODE = 2
function RS_CF_splitting(S,T)

	n = size(S,1)

    
	lambda = zeros(Int, n)

	Tp = T.colptr
	Tj = T.rowval
	Sp = S.colptr
	Sj = S.rowval

	interval_ptr = zeros(Int, n+1)
	interval_count = zeros(Int, n+1)
	index_to_node = zeros(Int,n)
	node_to_index = zeros(Int,n)

    for i = 1:n
		# compute lambda[i] - the number of nodes strongly coupled to node i
		lambda[i] = Sp[i+1] - Sp[i]
        interval_count[lambda[i] + 1] += 1
    end
	
	# initial interval_ptr
	@views accumulate!(+, interval_ptr[2:end], interval_count[1:end-1])

	# sort the nodes by the number of nodes strongly coupled to them:
	#   `index_to_node[idx]` is the node with `idx`th least number of nodes coupled to it
	#   `node_to_index[idx]` is the position of the `idx`th node in `index_to_node`
	# linear time and allocation-free equivalent to:
	#   sortperm!(index_to_node, lambda)
	#   node_to_index[index_to_node] .= 1:n
	interval_count .= 0 # temporarily zeroed, goes back to its original at end of loop
    for i = 1:n
        lambda_i = lambda[i] + 1
        interval_count[lambda_i] += 1
        index = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index] = i
        node_to_index[i]     = index
    end
	splitting = fill(U_NODE, n)

    # all nodes which no other nodes are strongly coupled to become F nodes
    for i = 1:n
		if lambda[i] == 0
            splitting[i] = F_NODE
		end
    end

	# i = index_to_node[top_index] can either refer to an F node or to the U node with the
	#	highest lambda[i].

	# index_to_node[interval_ptr[i]+1 : interval_ptr[i+1]] includes the set of U nodes with 
	#	i-1 nodes strongly coupled to them, and other "inactive" F and C nodes.
	
	# C nodes are always in index_to_node[top_index:n]. So at the end of the last 
	#	non-empty interval_ptr[i]+1 : interval_ptr[i+1] will be all the C nodes together 
	#	with some other inactive F nodes.
	
	# when lambda_k+1 > lambda_i, i.e. lambda_k == lambda_i,  where lambda_k+1 = lambda_i+1 
	#	is the new highest lambda[i], the line: `interval_ptr[lambda_k+1] = new_pos - 1`
	#	pushes the all the inactive C and F points to the end of the next now-non-empty 
	#	interval.
	for top_index = n:-1:1
		i = index_to_node[top_index]
		lambda_i = lambda[i] + 1
		interval_count[lambda_i] -= 1

		splitting[i] == F_NODE && continue

		@assert splitting[i] == U_NODE
		splitting[i] = C_NODE
		for j in nzrange(S,i)
			row = S.rowval[j]
			if splitting[row] == U_NODE
				splitting[row] = F_NODE

				# increment lambda for all U nodes that node `row` is strongly coupled to
				for k in nzrange(T, row)
					rowk = T.rowval[k]

					if splitting[rowk] == U_NODE
						# to ensure `intervalcount` is inbounds
						lambda[rowk] >= n - 1 && continue

						# move rowk to the end of its current interval
						lambda_k = lambda[rowk] + 1
						old_pos  = node_to_index[rowk]
						new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k]

						swap_node = index_to_node[new_pos]
						(index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, rowk)
						node_to_index[rowk] = new_pos
						node_to_index[swap_node] = old_pos

						# increment lambda[rowk]
						lambda[rowk] += 1

						# update intervals
						interval_count[lambda_k]   -= 1
						interval_count[lambda_k+1] += 1
						interval_ptr[lambda_k+1]    = new_pos - 1
					end
				end
			end
		end

		# decrement lambda for all U nodes that node i is strongly coupled to
		for j in nzrange(T,i)
			row = T.rowval[j]
			if splitting[row] == U_NODE
				# to ensure `intervalcount` is inbounds
				lambda[row] == 0 && continue

				# move row to the beginning of its current interval
				lambda_j = lambda[row] + 1
				old_pos  = node_to_index[row]
				new_pos  = interval_ptr[lambda_j] + 1

				swap_node = index_to_node[new_pos]
				(index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, row)
				node_to_index[row] = new_pos
				node_to_index[swap_node] = old_pos

				# decrement lambda[row]
				lambda[row] -= 1

				# update intervals
				interval_count[lambda_j]   -= 1
				interval_count[lambda_j-1] += 1
				interval_ptr[lambda_j]     += 1
			end
		end
	end
	splitting
end

#Step 3 Calculate the prolongation and restriction operator
#Find the index of non-zero elements in a vector
function find_nonzero(c)
    a = similar(c, Int)
    count = 1
    @inbounds for i in eachindex(c)
        a[count] = i
        count += (c[i] != zero(eltype(c)))
    end
    return resize!(a, count-1)
end
function neighborhood_positive(A,i)
    m,n=size(A)
    N = copy(A)
    N = tril(N,-1)+triu(N,1)
    for j = 1:n
        if N[i,j] > 0 
            N[i,j] = 1
        else
            N[i,j] = 0
        end
    end
    return find_nonzero(N[i,:])
end
function neighborhood_negative(A,i)
    m,n=size(A)
    N = copy(A)
    N = tril(N,-1)+triu(N,1)
    for j = 1:n
        if N[i,j] < 0 
            N[i,j] = 1
        else
            N[i,j] = 0
        end
    end
    return find_nonzero(N[i,:])
end



function prolongation(A,ϵ_AMG)
    #the size of prologation operator should be n_f*n_c
    S = strong_connection(A,ϵ_AMG)
    sparse_S = sparse(S)
    sparse_T = sparse(transpose(S))
    splitting = RS_CF_splitting(sparse_S,sparse_T)
    n_c = count(!iszero,splitting)
    n_f = size(splitting)[1]
    
   
    #Partition the C ∩ S_i
    S_positive = positive_couplings(A,ϵ_AMG)
    S_negative = negative_couplings(A,ϵ_AMG)
    C_set = find_nonzero(splitting)
    P = zeros((n_f,n_c))
    #if i in C, x_F=x_c
    P[C_set,1:n_c] .= I(n_c)
    #if i in F, based on (3.2) in meeting notes
    for i in findall(==(0), splitting)
        S_i_positive = intersect(C_set,find_nonzero(S_positive[i,:]))
        s_pi_size = size(S_i_positive)[1]
        S_i_negative = intersect(C_set,find_nonzero(S_negative[i,:]))
        s_ni_size = size(S_i_negative)[1]
        N_i_positive = neighborhood_positive(A,i)
        N_i_negative = neighborhood_negative(A,i)
        if isempty(S_i_positive) == false
            if isempty(N_i_positive) == false
                for k_p in N_i_positive
                    beta_numerator = sum(A[i,k_p])
                end
            else
                beta_numerator = 0
            end
            for j_p in S_i_positive
                beta_denominator = sum(A[i,j_p])
                beta_i = beta_numerator/beta_denominator
                j_p_s = findall(x -> x==j_p,S_i_positive)
                P[i,j_p_s] .= -beta_i*A[i,j_p]/A[i,i]
            end
        end
        if isempty(S_i_negative) == false
            if isempty(N_i_negative) == false
                for k_n in N_i_negative
                    alpha_numerator = sum(A[i,k_n])
                end
            else
                alpha_numerator = 0
            end
            for j_n in S_i_negative
                alpha_denominator = sum(A[i,j_n])
                alpha_i = alpha_numerator/alpha_denominator
                j_n_s = findall(x -> x==j_n,S_i_negative)
                P[i,j_n_s] .= -alpha_i*A[i,j_n]/A[i,i]
            end
        end
    end
    return P
end

function restriction(A,ϵ_AMG,σ)
    return σ*transpose(prolongation(A,ϵ_AMG))
end

function taylor(ww,xx,bb,dd,vv,ss)
    first_term = obj(ww,xx,bb,dd,vv)
    grad_v = obj_v(ww,xx,bb,dd,vv)
    grad_w = obj_w(ww,xx,bb,dd,vv)
    grad_b = obj_b(ww,xx,bb,dd,vv)
    grad_obj = grad_v/sqrt(norm(grad_v,Inf))+grad_b/sqrt(norm(grad_b,Inf))+grad_w/sqrt(norm(grad_w,Inf))
    second_term = grad_obj'*ss
    third_term = (ss'*matrix_A(ww,xx,bb,dd,vv)*ss)/2
    return first_term+second_term+third_term
end

#Levenberg Marquardt
function LM(ww,xx,bb,dd,vv,λ,ϵ)
    η_1=0.1
    η_2=0.75
    γ_1=0.85
    γ_2=0.5
    γ_3=1.5
    λ_min=10^(-6)
    θ=0.01
    grad_v = obj_v(ww,xx,bb,dd,vv)
    grad_w = obj_w(ww,xx,bb,dd,vv)
    grad_b = obj_b(ww,xx,bb,dd,vv)
    grad_obj = grad_v/sqrt(norm(grad_v,Inf))+grad_b/sqrt(norm(grad_b,Inf))+grad_w/sqrt(norm(grad_w,Inf))
    size_grad = size(grad_obj)[1]
    size_para = size(vv)[1]
    k=0
    if norm(grad_obj,2)>ϵ
        m_k = taylor(ww,xx,bb,dd,vv,ss)+λ*norm(ss,2)/2
        (x,stats) = cgls(matrix_A(grad_obj,-ones(size_grad),λ=λ,rtol=θ))
        ss=x
        ρ_numerator = obj(ww,xx,bb,dd,vv)-obj(ww+ss,xx,bb+ss,dd,vv+ss)
        ρ_denominator = taylor(ww,xx,bb,dd,vv,zeros(size_para))-taylor(ww,xx,bb,dd,vv,ss)
        ρ = ρ_numerator/ρ_denominator
        if ρ >= η_1
            ww = ww+ss
            vv = vv+ss
            bb = bb+ss
            if ρ>=η_2
                λ = max(λ_min,γ_2*λ)
            else
                λ = max(λ_min,γ_1*λ)
            end
        else
            ww=ww
            vv=vv
            bb=bb
            λ=γ_3*λ
        end
        k += 1
    else
    end
    return ww,vv,bb
end



















    
    





   




