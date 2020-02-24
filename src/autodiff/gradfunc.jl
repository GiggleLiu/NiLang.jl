export Grad, NGrad, Hessian, gradient

struct NGrad{N,FT} <: Function
    f::FT
end
function NGrad{N}(f::FT) where {N,FT}
    NGrad{N,FT}(f)
end

const Grad{FT} = NGrad{1,FT}
const Hessian{FT} = NGrad{2,FT}

Base.adjoint(f::Function) = Grad(f)
Base.adjoint(f::NGrad{N}) where {N} = NGrad{N+1}(f.f)
Base.show_function(io::IO, b::NGrad{N}, compact::Bool) where {N} = print(io, "$(b.f)"*"'"^N)
Base.show_function(io::IO, ::MIME"text/plain", b::NGrad{N}, compact::Bool) where {N} = print(io, b)
Base.display(bf::NGrad) = print(bf)
#Inv(f::NGrad{N}) where {N} = NGrad{N}(~f.f)
(_::Type{Inv{NGrad{N}}})(f::NGrad{M}) where {M, N} = NGrad{M-N}(f.f)
(_::Type{Inv{NGrad{M}}})(f::NGrad{M}) where {M} = f.f
#Grad(f::Inv) = Inv(f.f')


@i function (g::Grad)(args...; kwargs...)
    @safe @assert count(x -> x isa Loss, args) == 1
    @routine @invcheckoff begin
        iloss ← 0
        for i=1:length(args)
            if (tget(args,i) isa Loss, iloss==i)
                iloss += identity(i)
                (~Loss)(tget(args,i))
            end
        end
    end

    g.f(args...; kwargs...)
    GVar.(args)
    grad(tget(args,iloss)) += identity(1)
    (~g.f)(args...; kwargs...)

    ~@routine
end

function gradient(f, args; kwargs=())
    @assert count(x -> x isa Loss, args) == 1
    iloss = findfirst(x -> x isa Loss, args)
    _args = TupleTools.insertat(args, iloss, (args[iloss].x,))

    out = NiLangCore.wrap_tuple(GVar.(f(_args...; kwargs...)))
    _out = TupleTools.insertat(out, iloss, (chfield(out[iloss], grad, grad(out[iloss]) + 1),))
    grad.(NiLangCore.wrap_tuple((~f)(_out...; kwargs...)))
end
