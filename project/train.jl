using NiLang

@i function loss1(out!, x, y)
    y ⊕ x
    out! -= exp(x)
    out! += exp(y)
end

@i function loss2(out!, x, y)
    y ⊕ x
    out! -= exp(x)
    x -= log(-out!)
    out! += exp(y)
end

function train(lf)
    loss = Float64[]
    y = 1.6
    for i=1:100
        out!, x = 0.0, 0.3
        @instr lf(out!, x, y)
        push!(loss, out!)
        out! = 1.0
        @instr (~lf)(out!, x, y)
    end
    loss
end

res = train(loss2)
using DelimitedFiles
writedlm("train2.dat", res)
