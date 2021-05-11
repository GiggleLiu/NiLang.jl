function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(ROT),GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.08222882
    Base.precompile(Tuple{MinusEq{typeof(complex)},GVar{Float64, Float64},Union{GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}},Union{GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}}})   # time: 0.06912181
    Base.precompile(Tuple{typeof(wrap_bcastgrad),Type{AutoBcast{Int64, 4}},Vector{Float64}})   # time: 0.05026503
    Base.precompile(Tuple{Type{GVar},Matrix{Float64}})   # time: 0.044000912
    Base.precompile(Tuple{MinusEq{typeof(sincos)},Tuple{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{Float64, Float64}})   # time: 0.042384867
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, Float64},Any})   # time: 0.03920624
    Base.precompile(Tuple{typeof(NiLangCore.loaddata),Vector{Any},Vector{Any}})   # time: 0.03768901
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(-)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.036892496
    Base.precompile(Tuple{typeof(NiLangCore.loaddata),Type{Vector{Float64}},Vector{Float64}})   # time: 0.035732098
    Base.precompile(Tuple{MinusEq{typeof(log)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.03535623
    Base.precompile(Tuple{PlusEq{typeof(complex)},GVar{Float64, Float64},Union{GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}},Union{GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}}})   # time: 0.03526533
    Base.precompile(Tuple{typeof(grad),NTuple{4, GVar{Float64, Float64}}})   # time: 0.032369606
    Base.precompile(Tuple{typeof(wrap_jacobian),Type{AutoBcast{Float64, 1}},Vector{Float64}})   # time: 0.032170866
    Base.precompile(Tuple{Type{GVar},Vector{Float64}})   # time: 0.029704165
    Base.precompile(Tuple{typeof(IROT),GVar{Float64, Float64},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.025739137
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.024799693
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.023590498
    Base.precompile(Tuple{typeof(grad),Vector{GVar{Float64, Float64}}})   # time: 0.02317364
    Base.precompile(Tuple{typeof(grad),Tuple{Vector{GVar{Float64, Float64}}, Vector{GVar{Float64, Float64}}, Vector{GVar{Float64, Float64}}, GVar{Float64, Float64}}})   # time: 0.022934826
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},GVar{GVar{Float64, Float64}, GVar{Float64, Float64}},Int64})   # time: 0.022084313
    Base.precompile(Tuple{typeof(NiLangCore.loaddata),Vector{GVar{Float64, Float64}},Vector{Float64}})   # time: 0.02116515
    Base.precompile(Tuple{MinusEq{typeof(/)},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.020155573
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},Any,Float64})   # time: 0.016616046
    Base.precompile(Tuple{typeof(IROT),GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.016194412
    Base.precompile(Tuple{Type{GVar},Tuple{Float64, ComplexF64},Tuple{ComplexF64, ComplexF64}})   # time: 0.01619109
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{Float64, Float64},Any,Int64})   # time: 0.015741551
    Base.precompile(Tuple{Type{GVar},AbstractArray,AbstractArray})   # time: 0.013830282
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(inv)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.011192932
    Base.precompile(Tuple{typeof(wrap_jacobian),Type{AutoBcast{Float64, 3}},Vector{Float64}})   # time: 0.011075067
    Base.precompile(Tuple{typeof(wrap_jacobian),Type{AutoBcast{Float64, 2}},Vector{Float64}})   # time: 0.011053615
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(^)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}}})   # time: 0.010716249
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(complex)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.010689858
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(/)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010517918
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(*)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.01049392
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(-)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010452602
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(*)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.01041573
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(^)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}}})   # time: 0.010397522
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(+)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010353689
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(complex)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.010343891
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(/)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010335974
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(exp)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010250594
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(+)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.010101405
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(log)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.009795744
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(identity)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.009747697
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(exp)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.009696897
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(^)}, Union{GVar{Float64, Float64}, Float64}, Vararg{Any, N} where N}})   # time: 0.009630851
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(inv)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.009559213
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(log)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.009503618
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.009323083
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(identity)}, GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}, Complex{GVar{Float64, Float64}}}})   # time: 0.0093146
    Base.precompile(Tuple{MinusEq{typeof(cos)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.009097533
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(^)}, Union{GVar{Float64, Float64}, Float64}, Vararg{Any, N} where N}})   # time: 0.009068273
    Base.precompile(Tuple{typeof(grad),Tuple{MinusEq{typeof(complex)}, Union{GVar{Float64, Float64}, Float64}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.008931454
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{Float64, Float64},GVar,Int64})   # time: 0.008520359
    Base.precompile(Tuple{typeof(grad),Tuple{PlusEq{typeof(complex)}, Union{GVar{Float64, Float64}, Float64}, Complex{GVar{Float64, Float64}}, GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.00843281
    Base.precompile(Tuple{typeof(wrap_bcastgrad),Type{AutoBcast{Int64, 4}},Tuple{Float64}})   # time: 0.007901799
    Base.precompile(Tuple{typeof(grad),Tuple{typeof(NEG), GVar{Float64, Float64}, Complex{GVar{Float64, Float64}}}})   # time: 0.007657704
    Base.precompile(Tuple{PlusEq{typeof(/)},GVar{Float64, Float64},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.007580364
    Base.precompile(Tuple{typeof(grad),Tuple{GVar{Float64, Float64}, Matrix{GVar{Float64, Float64}}}})   # time: 0.007370161
    Base.precompile(Tuple{typeof(grad),Tuple{GVar{Float64, Float64}, GVar{Float64, Float64}, Vector{GVar{Float64, Float64}}}})   # time: 0.007349056
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, Float64},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.007118214
    Base.precompile(Tuple{PlusEq{typeof(identity)},Vector{GVar{Float64, Float64}},Vector{GVar{Float64, Float64}}})   # time: 0.007062924
    Base.precompile(Tuple{MinusEq{typeof(^)},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.007059243
    Base.precompile(Tuple{typeof(grad),Tuple{Float64, GVar{Float64, Float64}}})   # time: 0.006490685
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.005542749
    Base.precompile(Tuple{MinusEq{typeof(exp)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.004707843
    Base.precompile(Tuple{typeof(zero),GVar{Int64, AutoBcast{Int64, 3}}})   # time: 0.004680167
    Base.precompile(Tuple{typeof(grad),Tuple{GVar{Float64, Float64}, GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.00466041
    Base.precompile(Tuple{MinusEq{typeof(abs)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.004496061
    Base.precompile(Tuple{MinusEq{typeof(log)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.003723256
    Base.precompile(Tuple{PlusEq{typeof(/)},GVar{Float64, Float64},GVar,GVar{Float64, Float64}})   # time: 0.003611694
    Base.precompile(Tuple{typeof(grad),Tuple{GVar{Float64, Float64}, GVar{Float64, Float64}}})   # time: 0.003529446
    Base.precompile(Tuple{MinusEq{typeof(/)},GVar{Float64, Float64},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.003528577
    Base.precompile(Tuple{MinusEq{typeof(tan)},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.003438413
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, Float64},GVar})   # time: 0.003361321
    Base.precompile(Tuple{typeof(grad),Tuple{Float64, Float64}})   # time: 0.003360894
    Base.precompile(Tuple{Type{Inv{GVar}},AbstractArray})   # time: 0.003287998
    Base.precompile(Tuple{typeof(SWAP),GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.003202025
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.003030533
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, AutoBcast{Float64, 3}},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.002981777
    Base.precompile(Tuple{MinusEq{typeof(/)},GVar{Float64, Float64},GVar{Float64, Float64},Int64})   # time: 0.002951113
    Base.precompile(Tuple{MinusEq{typeof(log)},GVar{Float64, Float64},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.002914175
    Base.precompile(Tuple{Type{GVar},Float64,AutoBcast{Float64, _A} where _A})   # time: 0.002864071
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},GVar,Float64})   # time: 0.00271298
    Base.precompile(Tuple{PlusEq{typeof(/)},GVar{Float64, Float64},Real,GVar{Float64, Float64}})   # time: 0.002609891
    Base.precompile(Tuple{typeof(NiLangCore.loaddata),Type{GVar{Float64, AutoBcast{Float64, 3}}},Float64})   # time: 0.002460321
    Base.precompile(Tuple{typeof(-),GVar{Float64, AutoBcast{Float64, 1}}})   # time: 0.002408763
    Base.precompile(Tuple{typeof(NEG),GVar{Float64, AutoBcast{Float64, 1}}})   # time: 0.00236964
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{Float64, Float64},Float64,Int64})   # time: 0.002257135
    Base.precompile(Tuple{PlusEq{typeof(identity)},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.002224634
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},Real,Float64})   # time: 0.002007978
    Base.precompile(Tuple{SubConst{Float64},GVar{Float64, Float64}})   # time: 0.001776433
    Base.precompile(Tuple{PlusEq{typeof(^)},GVar{Float64, Float64},Real,Int64})   # time: 0.00169386
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, Float64},Float64,GVar{Float64, Float64}})   # time: 0.001672073
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, Float64},Int64,GVar{Float64, Float64}})   # time: 0.001642821
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},Float64,GVar{Float64, Float64}})   # time: 0.001626397
    Base.precompile(Tuple{AddConst{Float64},GVar{Float64, Float64}})   # time: 0.001618372
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},Int64,GVar{Float64, Float64}})   # time: 0.001491792
    Base.precompile(Tuple{PlusEq{typeof(*)},GVar{Float64, Float64},GVar{Float64, Float64},Float64})   # time: 0.001471461
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, Float64},GVar{Float64, Float64},Float64})   # time: 0.001428104
    Base.precompile(Tuple{Type{GVar},ComplexF64})   # time: 0.001348701
    Base.precompile(Tuple{MinusEq{typeof(cos)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001196095
    Base.precompile(Tuple{MinusEq{typeof(sin)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001177611
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001173775
    Base.precompile(Tuple{MinusEq{typeof(abs)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001170435
    Base.precompile(Tuple{typeof(chfield),Complex{GVar{Float64, Float64}},typeof(grad),ComplexF64})   # time: 0.00116481
    Base.precompile(Tuple{MinusEq{typeof(abs)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001158102
    Base.precompile(Tuple{MinusEq{typeof(exp)},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001156032
    Base.precompile(Tuple{MinusEq{typeof(*)},GVar{Float64, Float64},GVar{Float64, Float64}})   # time: 0.001134847
    Base.precompile(Tuple{MinusEq{typeof(sin)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001125385
    Base.precompile(Tuple{MinusEq{typeof(cos)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001105154
    Base.precompile(Tuple{MinusEq{typeof(exp)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001097152
    Base.precompile(Tuple{MinusEq{typeof(log)},GVar{Float64, AutoBcast{Float64, 2}}})   # time: 0.001096638
    Base.precompile(Tuple{MinusEq{typeof(/)},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.001095843
    Base.precompile(Tuple{MinusEq{typeof(^)},GVar{Float64, AutoBcast{Float64, 3}}})   # time: 0.001095656
    Base.precompile(Tuple{typeof(chfield),GVar{Float64, Float64},typeof(grad),Float64})   # time: 0.001003963
    Base.precompile(Tuple{MinusEq{typeof(identity)},GVar{Float64, Float64},Real})   # time: 0.00100021
end
