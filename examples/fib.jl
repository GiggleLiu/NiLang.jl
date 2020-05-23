# # Computing Fibonacci Numbers
# The following is an example that everyone likes, computing Fibonacci number recursively.
using NiLang

@i function rfib(out!, n::T) where T
    n1 ← zero(T)
    n2 ← zero(T)
    @routine begin
        n1 += identity(n)
        n1 -= identity(1)
        n2 += identity(n)
        n2 -= identity(2)
    end
    if (value(n) <= 2, ~)
        out! += identity(1)
    else
        rfib(out!, n1)
        rfib(out!, n2)
    end
    ~@routine
end

# The time complexity of this recursive algorithm is exponential to input `n`. It is also possible to write a reversible linear time with for loops.
# A slightly non-trivial task is computing the first Fibonacci number that greater or equal to a certain number `z`, where a `while` statement is required.

@i function rfibn(n!, z)
    @safe @assert n! == 0
    out ← 0
    rfib(out, n!)
    while (out < z, n! != 0)
        ~rfib(out, n!)
        n! += identity(1)
        rfib(out, n!)
    end
    ~rfib(out, n!)
end

# In this example, the postcondition `n!=0` in the `while` statement is false before entering the loop, and it becomes true in later iterations. In the reverse program, the `while` statement stops at `n==0`.
# If executed correctly, a user will see the following result.

rfib(0, 10)

# compute which the first Fibonacci number greater than 100.

rfibn(0, 100)

# and uncompute

(~rfibn)(rfibn(0, 100)...)

# This example shows how an addition postcondition provided by the user can help to reverse a control flow without caching controls.
