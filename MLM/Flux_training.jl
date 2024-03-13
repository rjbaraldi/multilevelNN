#in this file, we use Flux to make a reasonable initial guess for our problem
using Optimization, ProgressMeter, Flux
using Zygote, FluxOptTools, Statistics, Printf
function initial_guess(
    r,#the number of nodes in the hidden layer
    σ,#activation function 
    x,
    real_func,#real solution of PDEs
    learning_rate,
    n,#the maximum value of epoch
    m,#data dimension
    )
    if m == 2
        Hidden_Layer = Flux.Dense(2=>r,σ)
    else
        Hidden_Layer = Flux.Dense(1=>r,σ)
    end
    Output_Layer = Flux.Dense(r=>1,identity)
    Model = Flux.Chain(Hidden_Layer,Output_Layer)
    y = real_func.(x)
    #reshape the output vector
    yy = zeros(1,size(y)[1])
    for i in 1:size(x)[1]
        yy[1,i] = y[i]
    end
    optim = Flux.setup(Flux.Adam(learning_rate),Model)
    losses = []
    yhat = zeros(size(x)[1])
    @info "epoch    loss"
    @showprogress for epoch in 1:n
        loss,grads = Flux.withgradient(Model) do m
            yhat = m(x')
            Flux.Losses.mse(yhat,yy)
        end
        Flux.update!(optim,Model,grads[1])
        push!(losses,loss)
        (epoch %200 ==0) && @info @sprintf "%d  %8.3e" epoch losses[end] 
    end
    return Hidden_Layer.weight, Hidden_Layer.bias, vec(Output_Layer.weight), Output_Layer.bias
end