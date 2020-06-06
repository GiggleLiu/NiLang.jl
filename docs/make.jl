using Documenter, NiLang
using SparseArrays

using Literate
tutorialpath = joinpath(@__DIR__, "src/examples")
sourcepath = joinpath(@__DIR__, "../examples")
for jlfile in ["besselj.jl", "sparse.jl", "sharedwrite.jl", "qr.jl", "port_zygote.jl", "fib.jl", "unitary.jl", "nice.jl", "realnvp.jl", "boxmuller.jl"]
    Literate.markdown(joinpath(sourcepath, jlfile), tutorialpath)
end

makedocs(;
    modules=[NiLang],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "What and Why" => "why.md",
        "Tutorial" => Any[
            "tutorial.md",
            "examples/port_zygote.md",
           ],
        "Examples" => Any[
            "examples/fib.md",
            "examples/besselj.md",
            "examples/sparse.md",
            "examples/unitary.md",
            "examples/qr.md",
            "examples/nice.md",
            "examples/realnvp.md",
            "examples/boxmuller.md",
           ],
        "API & Manual" => Any[
            "instructions.md",
            "extend.md",
            "examples/sharedwrite.md",
            "api.md",
           ]
    ],
    repo="https://github.com/GiggleLiu/NiLang.jl/blob/{commit}{path}#L{line}",
    sitename="NiLang.jl",
    authors="JinGuo Liu, thautwarm",
)

deploydocs(;
    repo="github.com/GiggleLiu/NiLang.jl",
)
