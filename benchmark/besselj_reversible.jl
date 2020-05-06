using NiLang, NiLang.AD
using BenchmarkTools

@i function ifactorial(out!, n)
    INC(out!, INC)
    @invcheckoff for i=1:n
        mulint(out!, i)
    end
end

@i @inline function imul(out!, x, anc!)
    anc! += out! * x
    out! -= anc! / x
    SWAP(out!, anc!)
end

@i function ibesselj(out!, ν, z; atol=1e-8)
    @routine @invcheckoff begin
        k ← 0
        fact_nu ← zero(ν)
        halfz ← zero(z)
        halfz_power_nu ← zero(z)
        halfz_power_2 ← zero(z)
        out_anc ← zero(z)
        anc1 ← zero(z)
        anc2 ← zero(z)
        anc3 ← zero(z)
        anc4 ← zero(z)
        anc5 ← zero(z)

        halfz += z / 2
        halfz_power_nu += halfz ^ ν
        halfz_power_2 += halfz ^ 2
        ifactorial(fact_nu, ν)

        anc1 += halfz_power_nu/fact_nu
        out_anc += identity(anc1)
        while (abs(unwrap(anc1)) > atol && abs(unwrap(anc4)) < atol, k!=0)
            INC(k)
            @routine begin
                anc5 += identity(k)
                anc5 += identity(ν)
                anc2 -= k * anc5
                anc3 += halfz_power_2 / anc2
            end
            imul(anc1, anc3, anc4)
            out_anc += identity(anc1)
            ~@routine
        end
    end
    out! += identity(out_anc)
    ~@routine
end

# To test this function, we first define input parameters `a` and output `out!`
a = 1.0
out! = 0.0

# We wrap the output with a randomly initialized gradient field, suppose we get the gradients from a virtual loss function.
# Also, we need to initialize an empty gradient field for elements in input cuda tensor `a`.
out! = ibesselj(out!, 2, a)[1]
out_g! = GVar(out!, 1.0)
a_g = GVar(a)

# Call the inverse program, the multiple dispatch will drive you to the goal.
println("Benchmarking NiLang")
display(@benchmark ibesselj($out!, 2, $a))
println("Benchmarking NiLang.AD")
display(@benchmark (~ibesselj)($out_g!, 2, $a_g))
