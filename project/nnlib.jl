using NiLang, NiLang.AD

Base.zero(::Dup{T}) where T = Dup(zero(T))
Base.zero(::Type{Dup{T}}) where T = Dup(zero(T))
function Base.show(io::IO, obj::Bundle)
    vals = getproperty.(Ref(obj), fieldnames(typeof(obj)))
    print(io, "$(typeof(obj).name)$vals")
end

# the integrater
@i function leapfrog(field, x::Dup; Nt::Int, dt::Float64)
    @safe isreversible(field) || throw(InvertibilityError("Input function $f is not reversible."))
    # move ks for half step
    update_field(field, x.twin, x.x; dt=dt/2)
    for i=1:Nt
        update_field(field, x.x, x.twin; dt=dt)
        update_field(field, x.twin, x.x; dt=dt)
    end
    @safe @show field, x
end

@i function normal_logpdf(out, x::T, μ, σ) where T
    @anc anc1 = zero(T)
    @anc anc2 = zero(T)
    @anc anc3 = zero(T)

    @routine ri begin
        anc1 += x
        anc1 -= μ
        anc2 += anc1 / σ  # (x- μ)/σ
        anc3 += anc2 * anc2 # (x-μ)^2/σ^2
    end

    out -= anc3 * 0.5 # -(x-μ)^2/2σ^2
    out -= log(σ) # -(x-μ)^2/2σ^2 - log(σ)
    out -= log(2π)/2 # -(x-μ)^2/2σ^2 - log(σ) - log(2π)/2

    ~@routine ri
end

@i function normal_logpdf2d(out::T, x, μ, σ) where T
    @anc temp1 = zero(T)
    @anc temp2 = zero(T)
    normal_logpdf(temp1, x[1], μ[1], σ)
    normal_logpdf(out, x[2], μ[2], σ)
    out += temp1
    (~normal_logpdf)(temp1, x[1], μ[1], σ)
end

@i function ode_loss(μ, σ, field, xs::AbstractVector{TX}, loss_out::LT; Nt=Nt, dt=dt) where {LT,TX<:Dup}
    # backward run, drive target xs to source distribution space.
    # so that we can get `logp`.
    @anc anc_x = 0.0
    @safe println(loss_out)
    for i=1:length(xs)
        (~leapfrog)(field, xs[i]; Nt=Nt, dt=dt)
    end
    # then we evolve the `logp` to target space.
    PVar.(xs)
    for i=1:length(xs)
        anc_x += xs[i].x.x
        normal_logpdf(xs[i].x.logp, anc_x, μ, σ)
        anc_x -= xs[i].x.x
        leapfrog(field, xs[i]; Nt=Nt, dt=dt)
        # with logp, we compute log-likelihood
        loss_out += xs[i].x.logp / length(xs)
    end
end
