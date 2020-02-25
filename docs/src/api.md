```@meta
DocTestSetup = quote
    using NiLangCore, NiLang, NiLang.AD, Test
end
```

# API Manual
## Compiling Tools (Reexported from NiLangCore)
```@autodocs
Modules = [NiLangCore]
Order   = [:macro, :function, :type]
```

## Instructions
```@autodocs
Modules = [NiLang]
Order   = [:macro, :function, :type]
```

## Automatic Differentiation
```@autodocs
Modules = [NiLang.AD]
Order   = [:macro, :function, :type]
```
