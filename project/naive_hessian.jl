using NiLang, NiLang.AD

function (_::Type{Inv{GVar}})(x::GVar{<:GVar,<:GVar})
    Partial{:x}(x)
end

@i function ⊕(*)(out!::Partial, x::Partial, y::Partial)
    ⊕(*)(value(out!), value(x), value(y))
end

@i function ⊕(identity)(out!::Partial, x::Partial)
    ⊕(identity)(value(out!), value(x))
end

@i function hessian1(f, args; index::Int)
    @safe @assert count(x -> x isa Loss, args) == 1
    @anc iloss = 0
    @routine getiloss begin
        for i=1:length(args)
            if (args[i] isa Loss, iloss==i)
                iloss += identity(i)
            end
        end
    end

    # forward
    f'(args...)

    (~Loss)(args[iloss])
    for i = 1:length(args)
        GVar(grad(args[i]))
        GVar(value(args[i]))
    end
    grad(grad(args[index])) ⊕ 1.0
    # backward#2
    (Loss)(args[iloss])
    (~f')(args...)

    ~@routine getiloss
end

function hessian(f, args::Tuple)
    N = length(args)
    hmat = zeros(N, N)
    for i=1:N
        res = hessian1(f, args; index=i)
        @show res[2]
        hmat[:,i] .= map(x->x isa Loss ? grad(value(value(x))) :  grad(value(x)), res[2])
    end
    hmat
end


#hvar(x::GVar) = GVar(GVar(x.x), GVar(x.g))
#(_::Inv{typeof(hvar)})(x::GVar) = GVar((~GVar)(x.x), (~GVar)(x.g))

#Base.adjoint(gv::GVar) = GVar(gv.x', gv.g')
#Base.conj(gv::GVar) = GVar(conj(gv.x), conj(gv.g))
hessian(0.0, 2.0, 3.0)

hmat = zeros(3,3)
hessian1(⊕(*), (Loss(0.0), 2.0, 3.0); index=2)
hessian(⊕(*), (Loss(0.0), 2.0, 3.0))

# (x) -> (y, 1) -> (x, dL/dx=dL/dy*dy/dx)
