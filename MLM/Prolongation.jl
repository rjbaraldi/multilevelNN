export prolongation
#In this file, we give the prolongation,restriction operators and the C/F splitting.
using Optim, Flux
using Random,Printf
using LinearAlgebra, LinearOperators
using AlgebraicMultigrid, Krylov
using ForwardDiff, SparseArrays
#Define the prolongation and restriction operator
#Step 1 Define the matrix A as in meeting notes, grad_obj_1d_* function could be found in the main file
function matrix_A_1d(input_weights,input_biases,output_weights,output_bias,data,σ)
    ww_term = obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ)*obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ)'
    bb_term = obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ)*obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ)'
    vv_term = obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ)*obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ)'
    
    return ww_term/norm(obj_1d_approx_w(input_weights,input_biases,output_weights,output_bias,data,σ),Inf)+bb_term/norm(obj_1d_approx_b(input_weights,input_biases,output_weights,output_bias,data,σ),Inf)+vv_term/norm(obj_1d_approx_v(input_weights,input_biases,output_weights,output_bias,data,σ),Inf)
end

function matrix_A_2d(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    w1_grad = obj_2d_approx_w1(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    w2_grad = obj_2d_approx_w2(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    b_grad = obj_2d_approx_b(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    v_grad = obj_2d_approx_v(input_weights,input_biases,output_weights,output_bias,x1,x2,σ)
    w1_term = w1_grad*w1_grad'/norm(w1_grad,Inf)
    w2_term = w2_grad*w2_grad'/norm(w2_grad,Inf)
    b_term = b_grad*b_grad'/norm(b_grad,Inf)
    v_term = v_grad*v_grad'/norm(v_grad,Inf)
    return w1_term+w2_term+b_term+v_term
end

#Step 2 Define the classical strength of connection
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

function Si_positive(A,θ,i)
    m,n = size(A)
    T = copy(A)
    T = tril(T,-1)+triu(T,1)
    Sipositive = []
    for j in 1:n
        if T[i,j] != 0 && T[i,j] >= θ*abs(find_max_off_diag(T,i))
            push!(Sipositive,j)
        end
    end
    return Sipositive
end

function Si_negative(A,θ,i)
    m,n = size(A)
    T = copy(A)
    T = tril(T,-1)+triu(T,1)
    Sinegative = []
    for j in 1:n
        if T[i,j] != 0 && -T[i,j] >= θ*abs(find_max_off_diag(T,i))
             push!(Sinegative,j)
        end
    end
    return Sinegative
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
    
    #Step 3 Compute a C/F splitting using Ruge-Stuben coarsening
    #Parameters
    #----------
    #S: strength of connection matrix indicating the strength between nodes i and j (S[i,j])
    #----------
    #Returns
    #splitting: Array of length of S of ones (coarse) and zeros (fine) 
    ########
    #when we want to use RS_CF_splitting function to get the C/F-splitting, the input S is just from the strong_connection function, but we need to sparse it, and T is just the tranpose of S, which we also need to sparse it.
    #the following function RS_CF_splitting is cited from AlgebraicMultigrid package
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
#give the C/F splitting, where S,T should be sparse matrix and S could be obtained by strong_connection function and T is its transpose.
    function RS(S,T)
        remove_diag!(S)
        RS_CF_splitting(S,T)
    end

    #Step 4 Calculate the prolongation and restriction operator
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
        return find_nonzero(N[i,:])
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
        return find_nonzero(N[i,:])
    end
    
    
    
    function prolongation(A,ϵ_AMG)
        #the size of prologation operator should be n_f*n_c
        S = strong_connection(A,ϵ_AMG)
        sparse_S = sparse(S)
        sparse_T = sparse(transpose(S))
        splitting = RS(sparse_S,sparse_T) #splitting = AlgebraicMultigrid.RS_CF_splitting(sparse_S,sparse_T)
        n_c = count(!iszero,splitting)
        n_f = size(splitting)[1]
        
       
        #Partition the C ∩ S_i
        C_set = find_nonzero(splitting)
        P = zeros((n_f,n_c))
        #if i in C, x_F=x_c
        P[C_set,1:n_c] .= I(n_c)
        #if i in F, based on (3.2) in meeting notes
        for i in findall(==(0),splitting)
            S_i_positive = Si_positive(A,ϵ_AMG,i)
            #s_pi_size = size(S_i_positive)[1]
            S_i_negative = Si_negative(A,ϵ_AMG,i)
            #s_ni_size = size(S_i_negative)[1]
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
                for j_p in P_i_negative
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

