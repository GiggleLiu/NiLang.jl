using Documenter, NiLang

using Literate
tutorialpath = joinpath(@__DIR__, "src/examples")
for jlfile in ["besselj.jl"]
    Literate.markdown(joinpath("../examples", jlfile), tutorialpath)
end

makedocs(;
    modules=[NiLang],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Examples" => Any[
            "examples/besselj.md",
           ],
        "API Manual" => Any[
            "api.md",
           ]
    ],
    repo="https://github.com/GiggleLiu/NiLang.jl/blob/{commit}{path}#L{line}",
    sitename="NiLang.jl",
    authors="JinGuo Liu, thautwarm",
    assets=String[],
)

deploydocs(;
    repo="github.com/GiggleLiu/NiLang.jl",
)
