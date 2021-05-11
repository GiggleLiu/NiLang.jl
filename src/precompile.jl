const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{MinusEq{typeof(^)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},GVar{Float64, Float64}})   # time: 0.088883124
    Base.precompile(Tuple{PlusEq{typeof(^)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},GVar{Float64, Float64}})   # time: 0.08728426
    Base.precompile(Tuple{MinusEq{typeof(/)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.07497767
    Base.precompile(Tuple{PlusEq{typeof(exp)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.06419293
    Base.precompile(Tuple{PlusEq{typeof(/)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.041938376
    Base.precompile(Tuple{MinusEq{typeof(exp)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.04189861
    Base.precompile(Tuple{typeof(ROT),Float64,Float64,Irrational{:π}})   # time: 0.03481159
    Base.precompile(Tuple{MinusEq{typeof(exp)},ComplexF64,ComplexF64})   # time: 0.034116935
    Base.precompile(Tuple{PlusEq{typeof(exp)},ComplexF64,ComplexF64})   # time: 0.034112282
    Base.precompile(Tuple{typeof(PUSH!),Tuple{Float64, Float64, Vector{Float64}}})   # time: 0.031773232
    Base.precompile(Tuple{typeof(auto_expand),Expr})   # time: 0.031103289
    Base.precompile(Tuple{MinusEq{typeof(log)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.02943123
    Base.precompile(Tuple{PlusEq{typeof(log)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.028707368
    Base.precompile(Tuple{PlusEq{typeof(log)},ComplexF64,ComplexF64})   # time: 0.02753613
    Base.precompile(Tuple{MinusEq{typeof(inv)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.025810735
    Base.precompile(Tuple{PlusEq{typeof(inv)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.0257753
    Base.precompile(Tuple{PlusEq{typeof(^)},ComplexF64,ComplexF64,Float64})   # time: 0.019021787
    Base.precompile(Tuple{typeof(POP!),Vector{Any}})   # time: 0.01667686
    Base.precompile(Tuple{PlusEq{typeof(/)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.016225988
    Base.precompile(Tuple{typeof(COPYPOP!),Vector{Any}})   # time: 0.014254036
    Base.precompile(Tuple{MinusEq{typeof(/)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.014187833
    Base.precompile(Tuple{typeof(IROT),Float64,Float64,Irrational{:π}})   # time: 0.013347673
    Base.precompile(Tuple{typeof(auto_alloc),Expr})   # time: 0.011528118
    Base.precompile(Tuple{MinusEq{typeof(log)},ComplexF64,ComplexF64})   # time: 0.008786706
    Base.precompile(Tuple{typeof(COPYPUSH!),Tuple{Float64, Float64, Vector{Float64}}})   # time: 0.007922663
    Base.precompile(Tuple{PlusEq{typeof(inv)},ComplexF64,ComplexF64})   # time: 0.00758488
    Base.precompile(Tuple{typeof(NEG),Complex{GVar{Float64, Float64}}})   # time: 0.00560381
    Base.precompile(Tuple{typeof(POP!),Vector{Any},Vector{Float64}})   # time: 0.005197541
    Base.precompile(Tuple{PlusEq{typeof(*)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.004626822
    Base.precompile(Tuple{MinusEq{typeof(^)},ComplexF64,ComplexF64,Float64})   # time: 0.004009681
    Base.precompile(Tuple{typeof(NEG),ComplexF64})   # time: 0.002459327
    Base.precompile(Tuple{MinusEq{typeof(*)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.002440985
    Base.precompile(Tuple{typeof(PUSH!),Vector{Any}})   # time: 0.002278265
    Base.precompile(Tuple{typeof(PUSH!),Vector{Any},Vector{Int64}})   # time: 0.002102671
    Base.precompile(Tuple{PlusEq{typeof(+)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.001917077
    Base.precompile(Tuple{PlusEq{typeof(-)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.001910042
    Base.precompile(Tuple{MinusEq{typeof(*)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.001726165
    Base.precompile(Tuple{Type{BennettLog}})   # time: 0.001334291
    Base.precompile(Tuple{PlusEq{typeof(identity)},ComplexF64,ComplexF64})   # time: 0.001304064
    Base.precompile(Tuple{PlusEq{typeof(-)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.001289984
    Base.precompile(Tuple{MinusEq{typeof(+)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.001287298
    Base.precompile(Tuple{PlusEq{typeof(+)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.0012845
    Base.precompile(Tuple{MinusEq{typeof(-)},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}},Complex{GVar{Float64, Float64}}})   # time: 0.001269163
    Base.precompile(Tuple{typeof(-),NoGrad{Float64}})   # time: 0.001207526
    Base.precompile(Tuple{MinusEq{typeof(+)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.001151932
    Base.precompile(Tuple{MinusEq{typeof(inv)},ComplexF64,ComplexF64})   # time: 0.001151206
    Base.precompile(Tuple{MinusEq{typeof(-)},ComplexF64,ComplexF64,ComplexF64})   # time: 0.001148769
    Base.precompile(Tuple{typeof(chfield),ComplexF64,typeof(real),Float64})   # time: 0.001119287
    Base.precompile(Tuple{typeof(POP!),Tuple{Float64, GVar{Float64, Float64}, Vector{GVar{Float64, Float64}}}})   # time: 0.001055845
    Base.precompile(Tuple{PlusEq{typeof(complex)},Complex{GVar{Float64, Float64}},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.001005807
    Base.precompile(Tuple{PlusEq{typeof(complex)},ComplexF64,Float64,Float64})   # time: 0.001005075
    Base.precompile(Tuple{typeof(chfield),ComplexF64,typeof(imag),Float64})   # time: 0.001002244
end
