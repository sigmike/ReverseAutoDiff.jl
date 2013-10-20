module ReverseAutoDiff

type Acc
    v
    d
    parents
    ratios
end

Acc(x) = Acc(x, zero(x),(), ())

function *(a::Acc, b::Acc)
    Acc(a.v * b.v)
end

function backpropagate(v::Acc, r = one(v.v))
    v.d = r
    for i in 1:length(v.parents)
        backpropagate(v.parents[i], v.ratios[i])
    end
end

assign(a::Acc, x) = (a.d = 1; a.v = x)

function *(x, y::Acc)
    Acc(y.v * x, 1.0, (y,), (x,))
end

using Base.Test

function test()
    x = Acc(3.0)
    y = x
    backpropagate(y)
    @test y.v == 3.0
    @test x.d == 1.0

    assign(x, 2.0)
    y = x
    backpropagate(y)
    @test y.v == 2.0
    @test x.d == 1.0

    assign(x, 2.0)
    y = 3x
    backpropagate(y)
    @test y.v == 6.0
    @test x.d == 3.0
end

end
