using Documenter, NiLang

makedocs(;
    modules=[NiLang],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/GiggleLiu/NiLang.jl/blob/{commit}{path}#L{line}",
    sitename="NiLang.jl",
    authors="JinGuo Liu, thautwarm",
    assets=String[],
)

deploydocs(;
    repo="github.com/GiggleLiu/NiLang.jl",
)
