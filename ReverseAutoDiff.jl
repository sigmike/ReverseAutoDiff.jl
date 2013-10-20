module ReverseAutoDiff

type RADValue
    v
    d
end

RADValue(x) = RADValue(x, zero(x))

function *(a::RADValue, b::RADValue)
    RADValue(a.v * b.v)
end

function backpropagate(v::RADValue)
    v.d = 1.0
end

function test()
    x = RADValue(3.0)
    f() = x

    y = f()
    backpropagate(y)
    @assert isequal(y.v, 3.0)
    @assert isequal(x.d, 1.0)
end

end
