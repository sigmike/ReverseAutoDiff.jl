module ReverseAutoDiff

type Acc
    v
    d
end

Acc(x) = Acc(x, zero(x))

function *(a::Acc, b::Acc)
    Acc(a.v * b.v)
end

function backpropagate(v::Acc)
    v.d = 1.0
end

assign(a::Acc, x) = (a.v = x)

function test()
    x = Acc(3.0)
    y = x
    backpropagate(y)
    @assert isequal(y.v, 3.0)
    @assert isequal(x.d, 1.0)

    assign(x, 2.0)
    y = x
    backpropagate(y)
    @assert isequal(y.v, 2.0)
    @assert isequal(x.d, 1.0)
end

end
