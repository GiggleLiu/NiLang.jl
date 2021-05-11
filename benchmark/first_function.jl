t1 = time()
using NiLang

@i function dot(x, y, z)
    for i=1:10
        x += y[i]' * z[i]
    end
end
t2 = time()
println("costs $(t2-t1)s")
