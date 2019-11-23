@i function update_field(x, field_out, dt)
    xs += field_out * dt
end

struct PVar{T} <: Bundle{T}
    x::T
    logp::Float64
end
PVar(x) = PVar(x, 0.0)
(_::Inv{<:PVar})(x::PVar) = (@invcheck x.logp ≈ 0.0; x.x)

@i function update_field(x::T, y, dt) where T
    @anc field_out::T
    field(field_out, y, args...; kwargs...)
    x += field_out * dy
    (~field)(field_out, y, args...; kwargs...)
end

@i function update_field(x::PVar, y, field_out, dt)
    @anc field_out::T
    field(field_out, val(y), args...; kwargs...)
    # update x
    x += field_out * dt

    # update logp
    @routine grad begin
        GVar(Loss(field_out), y, args...)
        grad(field_out) += 1.0
        (~field)(field_out, y, args...; kwargs...)
    end
    x.logp -= grad(y) * dt
    ~@routine grad

    (~field)(field_out, val(y), args...; kwargs...)
end

@i function linear_field(field_out, x, θ)
    field_out += x * θ
end
