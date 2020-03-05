# Instruction Reference

## Instruction definitions

The Julia functions and symbols for instructions

| instruction | translated |   symbol   |
| ----------- | ---------- | ---- |
| $y \mathrel{+}= f(args...)$ | PlusEq(f)(args...) | $\oplus$ |
| $y \mathrel{-}= f(args...)$ | MinusEq(f)(args...) | $\ominus$ |
| $y \mathrel{\veebar}= f(args...)$ | \texttt{XorEq(f)(args...) | $\odot$ |

The list of reversible instructions that implemented in NiLang

| instruction | output   |
| ----------- | ---------- |
| ${\rm SWAP}(a, b)$ | $b, a$ |
| ${\rm ROT}(a, b, \theta)$ | $a \cos\theta - b\sin\theta, b \cos\theta + a\sin\theta, \theta$ |
| ${\rm IROT}(a, b, \theta)$ | $a \cos\theta + b\sin\theta, b \cos\theta - a\sin\theta, \theta$ |
| $y \mathrel{+}= a^\wedge b$ | $y+a^b, a, b$ |
| $y \mathrel{+}= \exp(x)$ | $y+e^x, x$ |
| $y \mathrel{+}= \log(x)$ | $y+\log x, x$ |
| $y \mathrel{+}= \sin(x)$ | $y+\sin x, x$ |
| $y \mathrel{+}= \cos(x)$ | $y+\cos x, x$ |
| $y \mathrel{+}= {\rm abs}(x)$ | $y+ |x|, x$ |
| ${\rm NEG}(y)$ | $-y$ |
| ${\rm CONJ}(y)$ | $y'$ |

"." is the broadcasting operations in Julia.

## Jacobians and Hessians for Instructions

See my [blog post](https://giggleliu.github.io/2020/01/18/jacobians.html).
