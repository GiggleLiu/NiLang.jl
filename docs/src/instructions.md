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
| ${\rm MULINT}(a, b)$ | $a * b, b$ |
| ${\rm DIVINT}(a, b)$ | $a / b, b$ |
| $y \mathrel{+}= a^\wedge b$ | $y+a^b, a, b$ |
| $y \mathrel{+}= \exp(x)$ | $y+e^x, x$ |
| $y \mathrel{+}= \log(x)$ | $y+\log x, x$ |
| $y \mathrel{+}= \sin(x)$ | $y+\sin x, x$ |
| $y \mathrel{+}= \cos(x)$ | $y+\cos x, x$ |
| $y \mathrel{+}= {\rm abs}(x)$ | $y+ |x|, x$ |
| ${\rm NEG}(y)$ | $-y$ |
| ${\rm CONJ}(y)$ | $y'$ |

"." is the broadcasting operations in Julia. The second argument of **MULINT** and **DIVINT** should be a nonzero integer.

#### Notes

1. What are **MULINT** and **DIVINT** reversible? The range of number representable by integer is much less than a floating point number, multiplying an integer to floating point number does not cause much rounding error, even less than floating point adder. However, multiplying or dividing two floating point numbers should not be considered reversible.


## Jacobians and Hessians for Instructions

See my [blog post](https://giggleliu.github.io/2020/01/18/jacobians.html).
