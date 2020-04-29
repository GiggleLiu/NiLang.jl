using Documenter, NiLang
using SparseArrays

using Literate
tutorialpath = joinpath(@__DIR__, "src/examples")
sourcepath = joinpath(@__DIR__, "../examples")
for jlfile in ["besselj.jl", "sparse.jl", "sharedwrite.jl"]
    Literate.markdown(joinpath(sourcepath, jlfile), tutorialpath)
end

makedocs(;
    modules=[NiLang],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "What and Why" => "why.md",
        "Examples" => Any[
            "examples/besselj.md",
            "examples/sparse.md",
            "examples/sharedwrite.md",
           ],
        "API & Manual" => Any[
            "instructions.md",
            "extend.md",
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
