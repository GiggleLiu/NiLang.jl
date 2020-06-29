export i_softmax_crossentropy, i_relu

function (_::PlusEq{typeof(argmax)})(out!, x::AbstractArray)
    out! += argmax(x)
    out!, x
end

function (_::MinusEq{typeof(argmax)})(out!, x::AbstractArray)
    out! -= argmax(x)
    out!, x
end


"""
    i_softmax_crossentropy(x, p, imax, xmax, Z, out)

Softmax-Cross entropy function.
"""
@i function i_softmax_crossentropy(x, p, imax, xmax, Z, out::T) where T
    logZ ← zero(T)
    yi ← zero(T)
    # subtract maximum
    imax += argmax(x)  # trade off space of xmax to time
    xmax += x[imax]
    # accumulate exp(x) to Z, and finally get logZ
    for i=1:length(x)
        x[i] -= xmax
        Z += Base.exp(x[i])
    end
    logZ += log(Z)
    for i=1:length(x)
        yi += logZ
        yi -= x[i]
        out += yi * p[i]
        yi += x[i]
        yi -= logZ
    end
    logZ -= log(Z)
end

"""
    i_relu(out!, x)

ReLU in machine learning.
"""
@i function i_relu(out!, x)
    @invcheckoff if (x > 0, ~)
        out! += identity(x)
    end
end
