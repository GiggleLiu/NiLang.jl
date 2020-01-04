import NiLang: Inv
import NiLang.AD: GVar

struct PVar{T,FLT} <: Bundle{T}
    x::T
    logp::FLT
end

PVar(x) = PVar(x, 0.0)
#PVar{T,FLT}(x) where {T,FLT} = PVar(T(x), zero(FLT))
#PVar{T,FLT}(x::Dup) where {T,FLT} = Dup(PVar{T,FLT}(x.x), PVar{T,FLT}(x.twin))
PVar(x::Dup) where T = Dup(PVar(x.x), PVar(x.twin))
(invg::Type{Inv{PVar}})(x::Dup) = Dup((~PVar)(x.x), (~PVar)(x.twin))

PVar{T,FLT}(x::PVar{T,FLT}) where {T,FLT} = x

GVar(x::PVar) = PVar(GVar(x.x), GVar(x.logp))
(invg::Type{Inv{GVar}})(x::PVar) = PVar(invg(x.x), invg(x.logp))

Base.zero(x::PVar) = PVar(zero(x.x))
Base.zero(x::Type{<:PVar{T}}) where T = PVar(zero(T))
Base.copy(x::PVar) = PVar(copy(x.x), copy(x.logp))

# inv kernel and field accesses
invkernel(x::PVar) = x.x
@fieldview NiLang.value(x::PVar) = x.x

# x += field(y) * dt
# TODO: support documentation
@i function update_field(field, x::T, y; dt) where T
    @anc field_out = zero(T)
    get_field(field, field_out, y)
    x += field_out * dt
    (~get_field)(field, field_out, y)
end

# x += field(y) * dt, and update logp.
# TODO: Prove ancilla gradients are discardable! it does not change reversibility.
# i.e. dg_var/dg_ancilla = 0, Beyesian deduction
@i function update_field(field, x::PVar{T}, y::PVar{T}; dt) where T
    @anc field_out = zero(T)
    get_field(field, field_out, y.x)
    # update x
    x.x += field_out * dt

    # update logp
    @routine grad begin
        GVar.((field, field_out, y))
        grad(field_out) ⊕ 1.0
        (~get_field)(field, field_out, y.x)
    end
    x.logp -= grad(y.x) * dt
    ~@routine grad

    (~get_field)(field, field_out, y.x)
end

abstract type Field end
struct LinearField{T} <: RevType
    θ::T
end
Base.zero(lf::LinearField{T}) where T = LinearField(zero(T))
Base.zero(::Type{LinearField{T}}) where T = LinearField(zero(T))
Base.isapprox(lf1::LinearField, lf2::LinearField; kwargs...) = isapprox(lf1.θ, lf2.θ; kwargs...)

NiLang.invkernel(lf::LinearField) = lf.θ

GVar(lf::LinearField) = LinearField(GVar(lf.θ))
(ig::Type{Inv{GVar}})(lf::LinearField) = LinearField((~GVar)(lf.θ))

@i function get_field(lf::LinearField, field_out, x)
    field_out += x * lf.θ
end
