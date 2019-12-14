include("nnlib.jl")
import NiLang: Inv
import NiLang.AD: GVar

struct PVar{T,FLT} <: Bundle{T}
    x::T
    logp::FLT
end
PVar(x) = PVar(x, 0.0)
PVar{T,FLT}(x) where {T,FLT} = PVar(T(x), zero(FLT))
PVar{T,FLT}(x::Dup) where {T,FLT} = Dup(PVar{T,FLT}(x.x), PVar{T,FLT}(x.twin))
PVar(x::Dup{T}) where T = PVar{T}(x)
PVar{T,FLT}(x::PVar{T,FLT}) where {T,FLT} = x
(_::Type{Inv{PVar}})(x::PVar) = (@invcheck val(x.logp) ≈ 0.0; x.x)
NiLangCore.isreversible(::PVar) = true
Base.zero(x::PVar) = PVar(zero(x.x))

GVar(x::Dup) = Dup(GVar(x.x), GVar(x.twin))
GVar(x::PVar) = PVar(GVar(x.x), GVar(x.logp))
(invg::Type{Inv{GVar}})(x::Dup) = Dup(invg(x.x), invg(x.twin))
(invg::Type{Inv{GVar}})(x::PVar) = PVar(invg(x.x), invg(x.logp))

# x += field(y) * dt
# TODO: support documentation
@i function update_field(field, x::T, y; dt) where T
    @anc field_out = zero(T)
    @safe @show(field, x, y)
    get_field(field, field_out, y)
    @safe @show(field, x, y)
    x += field_out * dt
    @safe @show(field, x, y)
    (~get_field)(field, field_out, y)
    @safe @show(field, field_out, y)
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
Base.zero(lf::LinearField{T}) where T = LinearField(T(0.0))
Base.zero(::Type{LinearField{T}}) where T = LinearField(T(0.0))

GVar(lf::LinearField) = LinearField(GVar(lf.θ))
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
