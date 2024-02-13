using Random, Flux, CUDA, Statistics
using AlgebraicMultigrid
using LinearAlgebra, Printf
using Plots

#using NLPModels, NLPModelsModifiers #might be handy for the actual recursive TR alg

Random.seed!(1234)
function acquire_pde_sol(x1, x2, n)
				#x = [x1; x2] in [0,1]
				d = LinRange(.1, 2.4, n) # should be [.01, .41,....2.4] if n = 6
				l = length(x1) #can also make these just a grid
				w = length(x2)

				u = reshape(zeros(Float32, l*w, n), (l, w, n))
				y = u 
				f = u 
				#probably a faster way to do this
				for k = 1:n
				  for i = 1:l
									for j = 1:w
													t1 = d[k]*cos(pi*x1[i])*cos(pi*x2[j]) #used a few times
													t2 = 5*cos(pi*x1[i]*x2[j])^2
													u[i, j, k] =  -(1 +  2*pi^2)*t1 -  t2*t1^3
													y[i, j, k] = -t1
													f[i, j, k] = -t1 + t2*(-t1)^3 #can consolidate
									end
					end
				end

				return u, y, f
end
function buildLap2D(x,y)
				nx = length(x)
				ny = length(y)

				D = diagm(-1=>ones(nx-1)) + diagm(0=>-2*ones(nx)) + diagm(1=>ones(nx-1))
				Id = I(nx)
        A = kron(D, Id) + kron(Id, D)
				A /= (nx+1)^2
				return A
end
function demo_nn()
  nx = 8
	x1 = LinRange(0.0, 1.0, nx+2)                                    # 50x1 Vector{Float32}
	x2 = x1
	nd = 6
	#get pde solution (2.1)
	u, y, f = acquire_pde_sol(x1, x2, nd)
	
	#construct 2d laplacian
	A = buildLap2D(x1,x2)
	# Define our model, a single-layer perceptron with one hidden layer:
	#W = rand((nx+2)^2, (nx+2)^2) #y is (nx+2)x(nx+2); A is (nx+2)^2x(nx+2)^2
	#b = rand((nx+2)^2,1)
	#model(x) = σ.(W*x .+ b) |> gpu        # move model to GPU, if available
	model = Dense((nx+2)^2=>(nx+2)^2, );
  function loss(yt, u, y) #in a sense, you're just learning A here
					 #should be single grid
					t = 1 #just one dataset right now
					(1/2*t)* sum((u .+ A*y .- model(y)).^2)	#take in vectorized inputs				
	end
  ##test gradients/jacobians
	#y0 = y[:,:, 1][:]
	#u0 = y[:, :, 1][:]
	#ps = Flux.params(model)
	#gradz = Flux.gradient(m-> loss(m(y0), u0, y0), model)
	#Jac = Zygote.jacobian(m->loss(m(x0), data), model) #for any arguments except Number and AbstractArray the result is nothing
	#Jac = Zygote.jacobian(m -> m(x0[:]), model)
  
	# To train the model, how to use batches:
	#(_, _, f) = data  
	#target = Flux.onehotbatch(f)                  
	#loader = Flux.DataLoader((u,y), target) |> gpu, batchsize=64, shuffle=true);
  
  optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
  
  # Training loop, using the whole data set 1000 times:
  losses = []
	@info "epoch    loss"
  for epoch in 1:10000
			for k=1:nd #there's a better way to structure their data if we use their optimizers
					ud = u[:, :, k][:]
					yd = y[:, :, k][:]
				  vals, grads = Flux.withgradient(model) do m
						yhat = ud .+ A*yd
						Flux.Losses.mse(m(yd), yhat)
					end
					Flux.update!(optim, model, grads[1])
          push!(losses, vals)  # logging, outside gradient
      end
			(epoch % 200 ==0) && @info @sprintf "%d  %8.3e" epoch losses[end] 
  end
  
  optim # parameters, momenta and output have all changed
	fout = zeros(size(f))
	for k = 1:nd
    ud = u[:, :, k][:]
		yd = y[:, :, k][:]
		fd = f[:, :, k][:] 

		fout[:,:,k] = reshape(model(yd), (nx+2, nx+2))  # first row is prob. of true, second row p(false)
		@info @sprintf "k:%d -->  ||u + Ay - f||= %8.8e    ||f - ftrue||=%8.8e" k norm(ud .+ A*yd -fd)  norm(reshape(fd, (nx+2, nx+2)) - fout[:,:,k])
	end

	Plots.surface(x1, x2,  f[:, :, 1], show=true) #camera = (-30, 30))
  
	Plots.surface(x1, x2, fout[:,:,1], show=true)
end

demo_nn()
