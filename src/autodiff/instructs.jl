# unary
@i @inline function Base.:-(a!::GVar)
    -(a!.x)
    -(a!.g)
end

@i @inline function DEC(a!::GVar)
    DEC(a!.x)
end

# +-
@i @inline function ⊖(identity)(a!::GVar, b::GVar)
    a!.x -= b.x
    b.g += a!.g
end
@nograd ⊖(identity)(a!::Real, b::GVar)
@nograd ⊖(identity)(a!::GVar, b::Real)

# +- (triple)
@i @inline function ⊖(+)(out!::GVar, x::GVar, y::GVar)
    out!.x -= x.x + y.x
    x.g += out! |> grad
    y.g += out! |> grad
end

@i @inline function ⊖(+)(out!::GVar, x::GVar, y::Real)
    out!.x -= (x |> value) + (y |> value)
    x.g += out! |> grad
end

@i @inline function ⊖(+)(out!::GVar, x::Real, y::GVar)
    out!.x -= (x |> value) + (y |> value)
    y.g += out! |> grad
end

@i @inline function ⊖(-)(out!::GVar, x::GVar, y::GVar)
    out!.x -= x.x - y.x
    x.g += out! |> grad
    y.g -= out! |> grad
end

@i @inline function ⊖(-)(out!::GVar, x::Real, y::GVar)
    out!.x -= (x |> value) - y.x
    y.g -= out!.g
end

@i @inline function ⊖(-)(out!::GVar, x::GVar, y::Real)
    out!.x -= x.x - (y |> value)
    x.g += out! |> grad
end

# NOTE: it will error on `SWAP(a!::GVar, b)` or `SWAP(a!, b:GVar)`
@i @inline function SWAP(a!::GVar, b!::GVar)
    SWAP(a! |> value,  b! |> value)
    SWAP(a!.g, b!.g)
end

# */
@i @inline function ⊖(*)(out!::GVar, x::GVar, y::GVar)
    out!.x -= x.x * y.x
    x.g += out!.g * y.x
    y.g += x.x * out!.g
end

@i @inline function ⊖(*)(out!::GVar, x::Real, y::GVar)
    out!.x -= (x |> value) * y.x
    y.g += (x |> value) * out!.g
end

@i @inline function ⊖(*)(out!::GVar, x::GVar, y::Real)
    out!.x -= x.x * (y |> value)
    x.g += out!.g * (y |> value)
end

for DIV in [:/, :÷]
@eval @i @inline function ⊖($DIV)(out!::GVar{T}, x::GVar, y::GVar) where T
    out!.x -= $DIV(x.x, y.x)
    @routine @invcheckoff begin
        a1 ← zero(out! |> grad)
        a2 ← zero(out! |> grad)
        a1 += x.x * out!.g
        a2 += $DIV(a1, y.x)
    end
    x.g += $DIV(out!.g, y.x)
    y.g -= $DIV(a2, y.x)
    ~@routine
end

@eval @i @inline function ⊖($DIV)(out!::GVar{T}, x::Real, y::GVar) where T
    out!.x -= $DIV(x, y.x)
    @routine @invcheckoff begin
        a1 ← zero(out!.g)
        a2 ← zero(out!.g)
        a1 += x * out!.g
        a2 += $DIV(a1, y.x)
    end
    y.g -= $DIV(a2, y.x)
    ~@routine
end

@eval @i @inline function ⊖($DIV)(out!::GVar, x::GVar, y::Real)
    out!.x -= $DIV(x.x, y)
    x.g += $DIV(out!.g, y)
end
end

@i @inline function ⊖(^)(out!::GVar{T}, x::GVar, n::GVar) where T
    ⊖(^)(out!.x, x.x, n.x)

    # grad x
    @routine @invcheckoff begin
        @zeros T anc1 anc2 anc3 jac1 jac2
        DEC(n.x)
        anc1 += x.x^n.x
        INC(n.x)
        jac1 += anc1 * n.x

        # get grad of n
        anc2 += log(x.x)
        anc3 += x.x ^ n.x
        jac2 += anc3*anc2
    end
    x.g += out!.g * jac1
    n.g += out!.g * jac2
    ~@routine
end

@i @inline function ⊖(^)(out!::GVar{T}, x::GVar, n::Real) where T
    ⊖(^)(out!.x, x.x, n)
    @routine @invcheckoff begin
        anc1 ← zero(x.x)
        jac ← zero(x.x)

        DEC(n |> value)
        anc1 += x.x ^ n
        INC(n |> value)
        jac += anc1 * n
    end
    x.g += out!.g * jac
    ~@routine
end

@i @inline function ⊖(^)(out!::GVar{T}, x::Real, n::GVar) where T
    ⊖(^)(out!.x, x, n.x)
    # get jac of n
    @routine @invcheckoff begin
        anc1 ← zero(x)
        anc2 ← zero(x)
        jac ← zero(x)

        anc1 += log(x)
        anc2 += x ^ n.x
        jac += anc1*anc2
    end
    n.g += out!.g * jac
    ~@routine
end

@i @inline function ⊖(atan)(out!::GVar{T}, y::GVar, x::GVar) where T
    ⊖(atan)(out!.x, y.x, x.x)
    @routine @invcheckoff begin
        @zeros T xy2 jac_x jac_y
        xy2 += abs2(x.x)
        xy2 += abs2(y.x)
        jac_y += x.x / xy2
        jac_x += (-y.x) / xy2
    end
    y.g += out!.g * jac_y
    x.g += out!.g * jac_x
    ~@routine
end

@i @inline function ⊖(atan)(out!::GVar{T}, x::GVar) where T
    ⊖(atan)(out!.x, x.x)
    @routine @invcheckoff begin
        xy2 ← one(T)
        xy2 += abs2(x.x)
    end
    x.g += out!.g / xy2
    ~@routine
end

@i @inline function ⊖(abs)(out!::GVar, x::GVar{T}) where T
    out!.x -= abs(x.x)
    if (x > 0, ~)
        x.g += out!.g
    else
        x.g -= out!.g
    end
end

@i @inline function ⊖(abs2)(out!::GVar, x::GVar{T}) where T
    out!.x -= abs2(x.x)
    x.g += out!.g * x.x
    x.g += out!.g * x.x
end
@nograd ⊖(abs2)(a!::GVar, b::Real)
@nograd ⊖(abs2)(a!::Real, b::GVar)

for op in [:*, :/, :^, :+, :-]
    @eval @nograd ⊖($op)(out!::GVar, x::Real, y::Real)
    @eval @nograd ⊖($op)(out!::Real, x::Real, y::GVar)
    @eval @nograd ⊖($op)(out!::Real, x::GVar, y::GVar)
    @eval @nograd ⊖($op)(out!::Real, x::GVar, y::Real)
end

@i @inline function ⊖(sqrt)(out!::GVar, x::GVar{T}) where T
    @routine @invcheckoff begin
        @zeros T anc1 anc2
        anc1 += sqrt(x.x)
        anc2 += 2 * anc1
    end
    out!.x -= anc1
    x.g += out!.g / anc2
    ~@routine
end

@i @inline function ⊖(exp)(out!::GVar, x::GVar{T}) where T
    @routine @invcheckoff begin
        anc1 ← zero(T)
        anc1 += exp(x.x)
    end
    out!.x -= anc1
    x.g += out!.g * anc1
    ~@routine
end

@i @inline function ⊖(log)(out!::GVar, x::GVar{T}) where T
    out!.x -= log(x.x)
    x.g += out!.g / x.x
end

@i @inline function ⊖(sin)(out!::GVar, x::GVar{T}) where T
    out!.x -= sin(x.x)
    @routine @invcheckoff begin
        anc1 ← zero(x.x)
        anc1 += cos(x.x)
    end
    x.g += out!.g * anc1
    ~@routine
end

@i @inline function ⊖(cos)(out!::GVar, x::GVar{T}) where T
    out!.x -= cos(x.x)
    @routine @invcheckoff begin
        anc1 ← zero(x.x)
        anc1 -= sin(x.x)
    end
    x.g += out!.g * anc1
    ~@routine
end

@i @inline function ⊖(tanh)(out!::GVar, x::GVar{T}) where T
    @routine @invcheckoff begin
        anc1 ← zero(x.x)
        anc2 ← one(x.x)
        anc1 += tanh(x.x)
        anc2 -= anc1^2
    end
    out!.x -= anc1
    x.g += out!.g * anc2
    ~@routine
end

@i @inline function ⊖(sincos)(out!::Tuple{T1,T1}, x::GVar{T}) where {T1<:GVar, T}
    @routine @invcheckoff begin
        s ← zero(T)
        c ← zero(T)
        (s, c) += sincos(x.x)
    end
    (out! .|> value) -= (s, c)
    x.g += (out! |> tget(1) |> grad) * c
    x.g -= (out! |> tget(2) |> grad) * s
    ~@routine
end

for op in [:sqrt, :exp, :log, :sin, :cos, :tanh]
    @eval @nograd ⊖($op)(out!::Real, x::GVar)
    @eval @nograd ⊖($op)(out!::GVar, x::Real)
end

@nograd ⊖(sincos)(out!::Tuple{<:Real,<:Real}, x::GVar)
@nograd ⊖(sincos)(out!::Tuple{<:GVar,<:GVar}, x::Real)

@i @inline function IROT(a!::GVar, b!::GVar, θ::GVar)
    IROT(a!.x, b!.x, θ.x)
    -(θ |> value)
    θ.x -= π/2
    ROT(a!.g, b!.g, θ.x)
    θ.g += a!.x * a!.g
    θ.g += b!.x * b!.g
    θ.x += π/2
    -(θ |> value)
    ROT(a!.g, b!.g, π/2)
end

@i @inline function IROT(a!::GVar, b!::GVar, θ::Real)
    IROT(a!.x, b!.x, θ)
    -(θ)
    θ -= π/2
    ROT(a!.g, b!.g, θ)
    θ += π/2
    -(θ)
    ROT(a!.g, b!.g, π/2)
end

@nograd IROT(a!::Real, b!::Real, θ::GVar)

export primitive_grad
function primitive_grad end

@i function (mf::MinusEq)(out!::GVar, args...; kwargs...)
    out!.x -= mf.f((args .|> value)...; kwargs...)
    (args .|> grad) .+= (@skip! out!.g) .* primitive_grad(mf.f, (args .|> value)...; kwargs...)
end

@i function (mf::MinusEq)(out!::GVar, x::GVar; kwargs...)
    out!.x -= mf.f(x .|> value; kwargs...)
    x.g += (@skip! out!.g) * primitive_grad(mf.f, x.x; kwargs...)
end

@i function :(-=)(convert)(out!::GVar{Tx, Tg}, y::GVar) where {Tx, Tg}
    out!.x -= convert(y.x)
    y.g += convert(out!.g)
end
