# NiLang.jl

NiLang is a reversible eDSL that can run backwards. The motation is to support source to source AD.

Check [our paper](https://arxiv.org/abs/2003.04617)!

Welcome for discussion in [Julia slack](https://slackinvite.julialang.org/), **#autodiff** and **#reversible-commputing** channel.

## Tutorials
```@contents
Pages = [
    "tutorial.md",
    "examples/port_zygote.md",
]
Depth = 1
```

Also see blog posts
* [How to write a program differentiably](https://nextjournal.com/giggle/how-to-write-a-program-differentiably)
* [Simulate a reversible Turing machine in 50 lines of code](https://nextjournal.com/giggle/rtm50)

## Examples
```@contents
Pages = [
    "examples/fib.md",
    "examples/besselj.md",
    "examples/sparse.md",
    "examples/unitary.md",
    "examples/qr.md",
    "examples/nice.md",
    "examples/realnvp.md",
    "examples/boxmuller.md",
]
Depth = 1
```

## Manual

```@contents
Pages = [
    "grammar.md",
    "instructions.md",
    "extend.md",
    "examples/sharedwrite.md",
    "api.md",
]
Depth = 2
```
