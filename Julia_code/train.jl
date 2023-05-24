
using ProgressBars
using EllipsisNotation
using Base:@kwdef
using StaticArrays
using BSON: @load, @save


function gm(N,p;a=-10, b=9)
    """
    Gaussian mixture p N(a, 1) + (1 - p)N(b, 1)
    """
    x,y = randn(N), rand(N).<p
    return x .+ b .+ (a-b).*y
end

g(x) = exp.(-0.5*abs.(x).^2)/√(2*π) # Gaussian density
σ(s) = s<=0 ? exp(s)/(1 + exp(s)) : 1/(1 + exp(-s)) # efficient sigmoid
dσ(s) = s<=0 ? exp(s) / (1 + exp(s))^2 : 1/(2+exp(s) + exp(-s)) # sigmoid derivative
σinv(p) = log(p/(1-p)) # sigmoid inverse

@kwdef struct Args 
    N_epochs::Int
    dt::Float32 = .01 # time discretization
    η::Float32 = .05 # GD step size (learning rate)
    N::Int = 1000
    K::Int = 1000
    stride::Int = 1000
end
    
abstract type AbstractSystem end

mutable struct OracleSystem <: AbstractSystem
    x #array
    n::Int
    s::Float32 
    p::Float32
    a::Float32
    b::Float32
    ∇
end

function OracleSystem(;n=1000, s=-0.9, a=-10., b=6.) # training data
    p=σ(s)
    x = gm(n,p; a=a, b=b)
    ∇ = ∇U(x, s, a, b)
    OracleSystem(x,n,s,p,a,b,∇)
end

mutable struct WeightedSystem <: AbstractSystem # walkers
    x #array
    w # weights (not exponential weights)
    n::Int
    s::Float32 
    p::Float32
    a::Float32
    b::Float32
    ∇
    δ
end

function WeightedSystem(;n=1000,s=0.1,a=-0.5,b=0.2)
    p = σ(s)
    x = gm(n,p; a=a, b=b)
    w = zeros(n)
    ∇ = ∇U(x,s,a,b)
    δ = zeros(1,3)
    WeightedSystem(x,w,n,s,p,a,b, ∇, δ) 
end


function U(x,s,a,b)
    """Returns - log ℙ(X=x) where X is a Gaussian mixture: p N(a,1) + (1-p)N(b,1), 
    where p = σ(s).  
    Beware of the sign: this is -log ℙ. """
    p = σ(s)
    prob = p*g(x.-a) .+ (1-p)*g(x.-b)
    return -log.(prob)
end

function ∂ₓU(x,s,a,b) #gradient in x
    """Returns only the gradient is x. """
    p = σ(s)
    ga, gb = g(x.-a), g(x.-b)
    numerator = p*ga .+  (1-p)*gb
    ∂x = (-p*(x.-a).*ga - (1-p)*(x.-b).*gb ) ./ numerator
    return -∂x
end

function ∇U(x,s,a,b) 
    """ ∇U where U = -log ℙ.  
    Beware of the sign: this is ∇(-log ℙ).   That is, we suppose that ℙ ∝ exp(-U). 
    Returns: ∂s, ∂a, ∂b, ∂x"""
    p = σ(s)
    ga, gb = g(x.-a), g(x.-b)
    numerator = p*ga .+  (1-p)*gb

    ∂s = dσ(s)* (ga - gb) ./ numerator
    ∂a = p*(x.-a).*ga ./ numerator
    ∂b = (1-p)*(x.-b).*gb ./ numerator

    return [-∂s -∂a -∂b]
end

∇U(S::AbstractSystem) = ∇U(S.x, S.s, S.a, S.b)


function diffuse!(x, gradient, dt)
    """One Langevin diffusion step with step size τ: 
    x ⟵ x - τ*gradient + 𝒩(0,2τ). """
    x[:] = x .- dt*gradient .+ √(2*dt).*randn(size(x)...)
    nothing
end

diffuse!(S::AbstractSystem, dt) = diffuse!(S.x, ∂ₓU(S.x, S.s, S.a, S.b), dt)

function create_system(;s₀=-0.9,N,K, η = 0.05, τ=0.01, weighted=true)
    @info "System is created"
    a₀,b₀ = -10, 6
    a,b = -10, 6
    s = 0.1
    x = gm(K,σ(s),a=a,b=b)

    #x = cat(randn(Int(K/2)).+a, randn(Int(K/2)).+b, dims=1)
    p = σ(s₀)
    x₀ = gm(N,p,a=a₀,b=b₀)
    if weighted
        w = zeros(K)
        moments = zeros(K, 3)
        student = ∇U(x,s,a,b)
        teacher = ∇U(x₀, s, a, b)
        return WeightedSystem(x, x₀, s, η, τ, a,b, a₀, b₀, s₀, w, student, teacher, 0., 0., 0., moments)
    else
        return System(x, x₀, 0.1, η, τ, a,b, a₀, b₀, s₀)
    end
end

function compute_q(S::WeightedSystem)::Float32
    W = exp.(S.w)
    q = sum( ( S.x .> 0 ) .* W ) / sum(W)
    return q
end

mutable struct History 
    s::Vector{Float32}
    p::Vector{Float32}
    a::Vector{Float32}
    b::Vector{Float32}
    δ::Matrix{Float32}
    w::Matrix{Float32}
    q::Vector{Float32}
    t
    args::Args
    qhat::Float32
    ahat::Float32
    bhat::Float32
    pzero::Float32
    oracle::OracleSystem
end

function create_logger(args::Args, T::OracleSystem)
    N_logged = args.N_epochs÷args.stride
    s = zeros(Float32, N_logged)
    p = zeros(Float32, N_logged)
    a = zeros(Float32, N_logged)
    b = zeros(Float32, N_logged)
    δ = zeros(Float32, N_logged, 3)
    w = zeros(Float32, N_logged, args.K)
    q = zeros(Float32, N_logged)
    t = 1:args.stride:args.N_epochs
    pzero = -1.0
    oracle = T
    return History(s,p,a,b,δ,w,q,t,args,0.0,0.0,0.0,pzero,oracle)
end

function log!(idx::Int, L::History, S::WeightedSystem)
    L.s[idx] = S.s
    L.p[idx] = S.p
    L.a[idx] = S.a
    L.b[idx] = S.b
    L.δ[idx, :] = S.δ
    L.w[idx, :] = S.w
    L.q[idx] = compute_q(S)
    nothing
end

function update!(
    S::WeightedSystem, 
    T::OracleSystem, 
    dt, 
    η, 
    update_ab, 
    update_jarczynski,
    )
    """Main function for the update of the system. 
    Args: 
    - dt time step 
    - η learning step
    - update_ab : if false, the modes are freezed
    - update_jarczynski : if false, the weights are set to 1
    """

    if update_jarczynski
        W = exp.(S.w)
        Z = sum(W)
    else 
        W = ones(S.n)
        Z = S.n 
    end

    teacher =  sum(T.∇, dims=1) / T.n
    student =  sum( S.∇ .* W, dims=1) / Z
    δ = dt * η * ( student - teacher )

    if !update_ab
        δ[2] = 0
        δ[3] = 0
    end 

    S.s += δ[1]
    S.a += δ[2]
    S.b += δ[3]
    S.p = σ(S.s)

    diffuse!(S, dt)
    ∇S = ∇U(S)
    ∇T = ∇U(T.x, S.s, S.a, S.b)

    if update_jarczynski
        S.w .-= 0.5 .* ( sum(S.∇ .* S.δ, dims=2) .+ sum(∇S .* δ, dims=2) )
    end

    S.δ = δ
    T.∇ = ∇T
    S.∇ = ∇S    

end


function train_and_log( # main training function. 
    N_epochs; 
    K=100, 
    N=100, 
    η=0.01, 
    dt=0.01, 
    stride=100, # logging steps
    s=1.4, 
    a=-12, 
    b=9,
    s_init=0.00, 
    a_init = -12, 
    b_init = 9, 
    update_ab=false,
    update_jarczynski=false,
    initialize_at_data=false,
    log_fname="logs/test.bson")

    T = OracleSystem(n=N, s=s, a=a, b=b)
    update_ab ? 
        S = WeightedSystem(n=K, s=s_init, b=b_init, a=a_init) : 
        S = WeightedSystem(n=K, s=s_init, b=b, a=a) 
    
        args = Args(N_epochs=N_epochs, dt=dt, η=η, N=N, K=K, stride=stride)

    if initialize_at_data
        @assert K==N "Number of walkers and training data must match for data-init PCD"
        S.x = T.x
        S.∇ = T.∇
    end

    logger = create_logger(args, T)
    logger.qhat = sum( (T.x .<= 0) ) / T.n
    logger.ahat = sum( (T.x).*(T.x .<= 0)) / sum( (T.x .<= 0) )
    logger.bhat = sum( (T.x).*(T.x .>= 0)) / sum( (T.x .>= 0) )
    logger.pzero = sum( S.x .< 0) / S.n

    epochs = ProgressBar(1:args.N_epochs)

    for t in epochs
        update!(S,T,args.dt,args.η, update_ab, update_jarczynski)
        t%stride == 1 ? log!(1+t÷stride, logger, S) : nothing 
    end
    
    @save log_fname logger
    return S
end


function main(n = 1000)
    @info "Jarzynski correction"
    train_and_log(
        n; K=200, N=200, stride=10, update_jarczynski=true, initialize_at_data = false, log_fname="./JAR.bson", s_init=0.001)
    @info "PCD"
    train_and_log(
        n; K=200, N=200, stride=10, update_jarczynski=false, initialize_at_data = true, log_fname="./PCD.bson", s_init=0.001)
    @info "Unweighted evolution"
    train_and_log(
        n; K=200, N=200, stride=10, update_jarczynski=false, initialize_at_data = false, log_fname="./UNW.bson", s_init=0.001)
end 

# launch the program : > julia train.jl
if abspath(PROGRAM_FILE) == @__FILE__
    main(1000000)
end


