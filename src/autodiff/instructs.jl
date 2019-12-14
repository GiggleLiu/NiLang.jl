@i function ⊖(exp)(out!::GVar, x::GVar{T}) where T
    value(out!) -= exp(value(x))
    @anc anc1 = zero(T)
    anc1 += exp(value(x))
    grad(x) += grad(out!) * anc1
    anc1 -= exp(value(x))
end

@i function ⊖(log)(out!::GVar, x::GVar{T}) where T
    value(out!) -= log(value(x))
    grad(x) += grad(out!) / value(x)
end

@i function ⊖(sin)(out!::GVar, x::GVar{T}) where T
    value(out!) -= sin(value(x))
    @anc anc1 = zero(T)
    anc1 += cos(value(x))
    grad(x) += grad(out!) * anc1
    anc1 -= cos(value(x))
end

@i function ⊖(cos)(out!::GVar, x::GVar{T}) where T
    value(out!) -= cos(value(x))
    @anc anc1 = zero(T)
    anc1 -= sin(value(x))
    grad(x) += grad(out!) * anc1
    anc1 += sin(value(x))
end

@i function IROT(a::GVar, b::GVar, θ::GVar)
    IROT(value(a), value(b), value(θ))
    NEG(value(θ))
    value(θ) ⊖ π/2
    ROT(grad(a), grad(b), value(θ))
    grad(θ) += value(a) * grad(a)
    grad(θ) += value(b) * grad(b)
    value(θ) ⊕ π/2
    NEG(value(θ))
    ROT(grad(a), grad(b), π/2)
end

#=
macro nograd(ex)
    @match ex begin
        :($f($(args...))) => :(
            @i function $f($(_render_type.(args)...))
            end
        )
    end
end

function _render_type(ex)
    @match ex begin
        :($x::$tp) => :($x::GVar{$tp})
        :($x) => :($x::GVar)
        _=>error("expect argument like `x::T` or `x`, got $ex")
    end
end

@nograd XOR(a!, b)
=#

# ugly patch
@i function ⊖(GVar{Float64,Float64})(a!::GVar, b::T) where T
    value(a!) ⊖ value(b)
end
