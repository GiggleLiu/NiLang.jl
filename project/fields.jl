include("nnlib.jl")
struct PVar{T} <: Bundle{T}
    x::T
    logp::Float64
end
PVar(x) = PVar(x, 0.0)
PVar{T}(x) where T = PVar(T(x), zero(T))
PVar{T}(x::Dup) where T = Dup(PVar{T}(x.x), PVar{T}(x.twin))
PVar(x::Dup{T}) where T = PVar{T}(x)
PVar{T}(x::PVar{T}) where T = x
(_::Inv{<:PVar})(x::PVar) = (@invcheck x.logp ≈ 0.0; x.x)
NiLangCore.isreversible(::PVar) = true
Base.zero(x::PVar) = PVar(zero(x.x))

# x += field(y) * dt
# TODO: support documentation
@i function update_field(field, x::T, y; dt) where T
    @anc field_out = zero(T)
    get_field(field, field_out, y)
    x += field_out * dt
    (~get_field)(field, field_out, y)
end

# x += field(y) * dt, and update logp.
@i function update_field(field, x::PVar{T}, y; dt) where T
    @anc field_out = zero(T)
    get_field(field, field_out, val(y))
    # update x
    x += field_out * dt

    # update logp
    @routine grad begin
        Loss(field_out)
        GVar.((field, field_out, y))
        grad(field_out) += 1.0
        (~get_field)(field, field_out, y)
    end
    x.logp -= grad(y) * dt
    ~@routine grad

    (~get_field)(field, field_out, val(y))
end

abstract type Field end
struct LinearField{T}
    θ::T
end

NiLang.AD.GVar(lf::LinearField) = LinearField(GVar(lf.θ))
(ig::Type{Inv{GVar}})(lf::LinearField) = LinearField((~GVar)(lf.θ))
NiLangCore.isreversible(::LinearField) = true

Base.:~(lf::LinearField) = lf.θ
function NiLangCore.chfield(x::T, ::Type{<:LinearField}, lf::LinearField{T}) where T
    lf.θ
end
Base.isapprox(lf1::LinearField, lf2::LinearField; kwargs...) = isapprox(lf1.θ, lf2.θ; kwargs...)

@i function get_field(lf::LinearField, field_out, x)
    field_out += x * lf.θ
end
