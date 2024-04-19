export prolongation
#In this file, we give the prolongation,restriction operators and the C/F with respect to the known A
using Optim, Flux
using Random,Printf
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
using ForwardDiff, SparseArrays
#priori 
abstract type strength end
struct classical_negative{T} <: strength
    θ::T
end
classical_negative(;θ = 0.9) = classical_negative(θ)

function (c::classical_negative)(At::SparseMatrixCSC{Tv,Ti}) where {Ti,Tv}

    θ = c.θ

    m, n = size(At)
    T = copy(At)

    for i = 1:n
        _m = find_max_negative_off_diag(T, i)
        threshold = θ * _m
        for j in nzrange(T, i)
            row = T.rowval[j]
            val = T.nzval[j]

            if row != i
                if -val >= threshold
                    T.nzval[j] = val
                else
                    T.nzval[j] = 0
                end
            end

        end
    end
    
    dropzeros!(T)

    scale_cols_by_largest_negative_entry!(T)

    -adjoint(T), -T
end

function find_max_negative_off_diag(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        if row != i
            m = max(m, -val)
        end
    end
    m
end

function find_negative_max(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        m = max(m, -val)
    end
    m
end

function scale_cols_by_largest_negative_entry!(A::SparseMatrixCSC)
    n = size(A, 1)
    for i = 1:n
        _m = find_negative_max(A, i)
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end
    A
end

const F_NODE = 0
const C_NODE = 1
const U_NODE = 2

struct RS_priori
end

function remove_diag!(a)
	n = size(a, 1)
	for i = 1:n
		for j in nzrange(a, i)
			if a.rowval[j] == i
				a.nzval[j] = 0
			end
   		end
	end
	dropzeros!(a)
end

function (::RS_priori)(S,T)
	remove_diag!(S)
	RS_CF_splitting_priori(S, T)
end

function RS_CF_splitting_priori(S::SparseMatrixCSC, T::SparseMatrixCSC)

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
		for j in nzrange(S, i)
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
		for j in nzrange(T, i)
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


#posteriori
function A_poster(A,splitting)
    A_copy = copy(A)
    C_sets = findall(x -> x != 0, splitting)
    for i in C_sets
        A_copy[i,:] .= 0
    end
    return A_copy
end

struct classical_positive{T} <: strength
    θ::T
end
classical_positive(;θ = 0.9) = classical_positive(θ)

function (c::classical_positive)(At::SparseMatrixCSC{Tv,Ti}) where {Ti,Tv}

    θ = c.θ

    m, n = size(At)
    T = copy(At)

    for i = 1:n
        _m = find_max_abs_off_diag(T, i)
        threshold = θ * _m
        for j in nzrange(T, i)
            row = T.rowval[j]
            val = T.nzval[j]

            if row != i
                if val >= threshold
                    T.nzval[j] = val
                else
                    T.nzval[j] = 0
                end
            end

        end
    end
    
    dropzeros!(T)

    scale_cols_by_largest_negative_entry!(T)

    adjoint(T), T
end

function find_max_abs_off_diag(A, i)
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

function find_abs_max(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        m = max(m, abs(val))
    end
    m
end

function scale_cols_by_largest_abs_entry!(A::SparseMatrixCSC)
    n = size(A, 1)
    for i = 1:n
        _m = find_abs_max(A, i)
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end
    A
end



struct RS_posteriori
end



function (::RS_posteriori)(S,T)
	remove_diag!(S)
	RS_CF_splitting_posteriori(S, T)
end

function RS_CF_splitting_posteriori(S::SparseMatrixCSC, T::SparseMatrixCSC)

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
		for j in nzrange(S, i)
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
		for j in nzrange(T, i)
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

function splitting_A(A,θ)
    A_s = sparse(A)
    strength_A_priori = classical_negative(θ)
    S,T = strength_A_priori(A_s)
    S = sparse(S)
    T = sparse(T)
    CF = RS_priori()
    splitting_priori = CF(S,T)
    strength_A_posteriori = classical_positive(θ)
    Sp, Tp = strength_A_posteriori(A_poster(A_s,splitting_priori))
    Sp = sparse(Sp)
    Tp = sparse(Tp)
    CF_posteriori = RS_posteriori()
    splitting_posteriori = CF_posteriori(Sp,Tp)
    return splitting_posteriori+splitting_priori
end

A = [-5.5 -6.0 3.0 -10.0 -9.8 5.0 4.0 2.0 1.0 0.0;
     -6.0 4.0 0.0 -7.0 -6.5 3.0 0.0 -2.0 0.0 1.0;
     3.0 0.0 -4.5 -6.5 -7.0 0.0 4.0 1.0 0.0 0.0;
     -10.0 -7.0 -6.5 2.0 -9.2 6.0 5.0 9.8 10.0 10.0;
     -9.8 -6.5 -7.0 -9.2 4.0 0.0 2.0 10.0 -10.0 1.0;
     5.0 3.0 0.0 6.0 0.0 5.5 -5.0 -4.8 7.0 8.0;
     4.0 0.0 4.0 5.0 2.0 -5.0 1.0 0.0 4.0 -4.9;
     2.0 -2.0 1.0 9.8 10.0 -4.8 0.0 2.0 -5.0 7.0;
     1.0 0.0 0.0 10.0 -10.0 7.0 4.0 -5.0 4.8 -9.9;
     0.0 1.0 0.0 10.0 1.0 8.0 -4.9 7.0 -9.9 5]

function neighborhood_positive(A,i)
    n=size(A)[2]
    N = copy(A)
    N = tril(N,-1)+triu(N,1)
    for j = 1:n
        if N[i,j] > 0 
            N[i,j] = 1
        else
            N[i,j] = 0
        end
    end
    return findall(x -> x != 0, N[i,:])
end
    
function neighborhood_negative(A,i)
    n=size(A)[2]
    N = copy(A)
    N = tril(N,-1)+triu(N,1)
    for j = 1:n
        if N[i,j] < 0 
            N[i,j] = 1
        else
            N[i,j] = 0
        end
    end
    return findall(x -> x != 0, N[i,:])
end
    
function Strong_negative(A,θ,i)
    m,n = size(A)
    A = sparse(A)
    T = copy(A)
    T = tril(T,-1)+triu(T,1)
    Sinegative = []
    for j in 1:n
        if T[i,j] != 0 && -T[i,j] >= θ*find_max_negative_off_diag(T,i)
            push!(Sinegative,j)
        end
    end
    return Sinegative
end  

function Strong_positive(A,θ,i)
    m,n = size(A)
    A = sparse(A)
    T = copy(A)
    T = tril(T,-1)+triu(T,1)
    Sipositive = []
    for j in 1:n
        if T[i,j] != 0 && T[i,j] >= θ*find_max_abs_off_diag(T,i)
            push!(Sipositive,j)
        end
    end
    return Sipositive
end  

function prolongation(A,ϵ_AMG)
        
    #the size of prologation operator should be n_f*n_c
    splitting = splitting_A(A,ϵ_AMG)#splitting = AlgebraicMultigrid.RS_CF_splitting(sparse_S,sparse_T)
    n_c = count(!iszero,splitting)
    n_f = size(splitting)[1]
        
       
    #Partition the C ∩ S_i
    C_set = findall(x -> x != 0,splitting)
    P = zeros((n_f,n_c))
    #if i in C, x_F=x_c
    P[C_set,1:n_c] .= I(n_c)
    #if i in F, based on (3.2) in meeting notes
    for i in findall(==(0),splitting)
        S_i_positive = Strong_positive(A,ϵ_AMG,i)
        S_i_negative = Strong_negative(A,ϵ_AMG,i)
        N_i_positive = neighborhood_positive(A,i)
        N_i_negative = neighborhood_negative(A,i)
        P_i_positive = intersect(C_set,S_i_positive)
        P_i_negative = intersect(C_set,S_i_negative)
        if isempty(P_i_positive) == false
            if isempty(N_i_positive) == false
                beta_numerator = 0
                for k_p in N_i_positive
                    beta_numerator += A[i,k_p]
                end
            else
                beta_numerator = 0
            end
            beta_denominator = 0
            for j_p in P_i_positive
                beta_denominator += A[i,j_p]
            end
            beta_i = beta_numerator/beta_denominator
            for j_p in P_i_positive
                j_p_s = findall(x -> x==j_p,C_set)
                P[i,j_p_s] .= -beta_i*A[i,j_p]/A[i,i]
            end
        end
        if isempty(P_i_negative) == false
            if isempty(N_i_negative) == false
                alpha_numerator = 0
                for k_n in N_i_negative
                    alpha_numerator += A[i,k_n]
                end
            else
                alpha_numerator = 0
            end
            alpha_denominator = 0
            for j_n in P_i_negative
                alpha_denominator += A[i,j_n]
            end
            alpha_i = alpha_numerator/alpha_denominator
            for j_n in P_i_negative    
            j_n_s = findall(x -> x==j_n,C_set)
            P[i,j_n_s] .= -alpha_i*A[i,j_n]/A[i,i]
            end
        end
    end
    return P
end
    
function restriction(A,ϵ_AMG,σ_R)
    return σ_R*transpose(prolongation(A,ϵ_AMG))
end
