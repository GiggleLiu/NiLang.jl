using NiLang, NiLang.AD

"""A field function is called as `f(field_out, x, params...), it should be reversible.`"""
struct LeapFrog{FT} <: Function
    f::FT
    function LeapFrog(f::FT) where FT
        isreversible(f) || throw(InvertibilityError("Input function $f is not reversible."))
        new{FT}(f)
    end
end
Base.zero(::Dup{T}) where T = Dup(zero(T))

# the integrater
@i function (lf::LeapFrog)(x, params; Nt::Int, dt::Float64)
    # move ks for half step
    @anc fout::Float64
    Dup(x)
    lf.f(fout, x.x, params...)
    x.twin += fout * (dt/2)
    (~lf.f)(fout, x.x, params...)
    for i=1:Nt
        lf.f(fout, x.twin, params...)
        x.x += fout * dt
        (~lf.f)(fout, x.twin, params...)

        lf.f(fout, x.x, params...)
        x.twin += fout * dt
        (~lf.f)(fout, x.x, params...)
    end
end

@i function normal_logpdf(out, x::T, μ, σ) where T
    @anc anc1::T
    @anc anc2::T
    @anc anc3::T

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
    @anc temp1::T
    @anc temp2::T
    normal_logpdf(temp1, x[1], μ[1], σ)
    normal_logpdf(out, x[2], μ[2], σ)
    out += temp1
    (~normal_logpdf)(temp1, x[1], μ[1], σ)
end

using Test

#=
using DelimitedFiles
using UnicodePlots

@i function ode_loss!(μ, σ, logp, target_xs, ks, field, ode!, logpdf, jacobian_out, field_out, loss_out, θ, Nt, dt)
    @anc loss_temp = 0.0
    # backward run, drive target xs to source distribution space.
    # so that we can get `logp`.
    ~ode!(target_xs, ks, nothing, field, field_out, θ, Nt, dt)
    # then we evolve the `logp` to target space.
    logpdf.(logp, target_xs, Ref(μ), Ref(σ))

    ode!(target_xs, ks, logp, field, field_out, θ, Nt, dt)
    # with logp, we compute log-likelihood
    add!.(loss_temp, logp)
    infer!(*, loss_out, loss_temp, 1/length(target_xs))
    ~(add!.(loss_temp, logp))
end

function analyze_flow(xs0, logp0, xs, logp; xlim=[-5,5], ylim=[0,1])
    order = sortperm(xs0)[1:10:end]
    lineplot(xs0[order], exp.(logp0[order]); xlim=xlim, ylim=ylim,title="p(x), before transformation") |> println

    fy = fit(Histogram, xs, range(xlim..., length=200), closed=:left)
    xticks = (fy.edges[1][1:end-1] .+ fy.edges[1][2:end]) ./ 2
    lineplot(xticks, fy.weights; xlim=xlim, title="data, after transformation") |> println

    order = sortperm(xs)[1:10:end]
    lineplot(xs[order], exp.(logp[order]); xlim=xlim, ylim=ylim,title="p(x), after transformation") |> println
end

function forward_sample(μ, σ, field, θ; nsample=100, Nt=100, dt=0.01, xlim=[-5,5], ylim=[0,1])
    # source distribution
    source_distri = Normal(μ, σ)
    xs0 = rand(source_distri, nsample)
    logp0 = max.(logpdf.(source_distri, xs0), -10000)

    # solving the ode, and obtain the probability change
    @newvar v_xs = copy(xs0)
    @newvar v_ks = copy(xs0)
    @newvar v_logp = copy(logp0)
    @newvar v_θ = θ
    @newvar field_out = 0.0
    tape = ode!(v_xs, v_ks, v_logp, field, field_out, v_θ, Nt, dt)

    play!(tape)
    analyze_flow(xs0, logp0, v_xs[], v_logp[]; xlim=xlim)
end

function get_loss_gradient!(forward, loss_out, θ)
    reset_grad!(forward)
    resetreg(forward)
    play!(forward)

    ll = loss_out[]

    grad(loss_out)[] = 1
    play!(forward')
    return ll, grad(θ)
end

function train(xs_target, θ; niter=100, Nt=100, dt=0.01, lr=0.1)
    μ = 0.0
    σ = 1.0
    @newvar loss_out = 0.0
    @newvar logp_out = zeros(Float64, length(xs_target))
    @newvar v_xst = xs_target
    @newvar v_kst = copy(xs_target)
    @newvar field_out = 0.0
    @newvar jacobian_out = zero(logp_out[])
    forward = ode_loss!(μ, σ, logp_out, v_xst, v_kst, field, ode!,
        inv_normal_logpdf!, jacobian_out, field_out, loss_out, θ, Nt, dt)

    init_grad!(forward)
    local ll
    for i=1:niter
        logp_out[] .= 0.0
        loss_out[] = 0.0
        ll, θδ = get_loss_gradient!(forward, loss_out, θ)
        println("Step $i, log-likelihood = $ll")
        θ[] += θδ[] * lr
        @show θ[]
    end
    return ll, forward
end

export softmax_cross_entropy!
@invfunc softmax_cross_entropy!(x, p, imax, xmax, Z, out) begin
    @ancilla logZ::Float64
    @ancilla i::Int
    @ancilla N::Int
    # subtract maximum
    infer!(argmax, imax, x)  # trade off space of xmax to time
    add!(xmax, x[imax])
    sub!.(x, xmax)

    # accumulate exp(x) to Z, and finally get logZ
    infer!.(exp, Z, x)
    infer!(log, logZ, Z)
    sub!.(x, logZ)
    neg!.(x)
    ~infer!(log, logZ, Z)

    infer!(length, N, x)
    infer!.(*, out, x, p)
    ~infer!(length, N, x)
end
=#
