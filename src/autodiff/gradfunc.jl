export Grad, NGrad, Hessian

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
    iloss ← 0
    @routine for i=1:length(args)
        if (tget(args,i) isa Loss, iloss==i)
            iloss += identity(i)
            (~Loss)(tget(args,i))
        end
    end

    g.f(args...; kwargs...)
    GVar.(args)
    grad(tget(args,iloss)) += identity(1)
    (~g.f)(args...; kwargs...)

    ~@routine
end
