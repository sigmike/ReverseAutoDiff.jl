module ReverseAutoDiff

type Acc
    v
    parents
    ratios
    d

end

Acc(x, parents, ratios) = Acc(x, parents, ratios, zero(x))
Acc(x) = Acc(x, (), ())

function *(a::Acc, b::Acc)
    Acc(a.v * b.v)
end

function backpropagate(v::Acc, r = one(v.v))
    v.d = r
    for i in 1:length(v.parents)
        backpropagate(v.parents[i], v.d * v.ratios[i])
    end
end

assign(a::Acc, x) = (a.d = 1; a.v = x)

function *(x, y::Acc)
    Acc(y.v * x, (y,), (x,))
end

function +(x::Acc, y::Acc)
    Acc(x.v + y.v, (x,y), (1,1))
end

function ==(x::Acc, y)
    x.v == y
end

using Base.Test

function test()
    x = Acc(3.0)
    y = x
    backpropagate(y)
    @test y == 3.0
    @test x.d == 1.0

    assign(x, 2.0)
    y = x
    backpropagate(y)
    @test y == 2.0
    @test x.d == 1.0

    assign(x, 2.0)
    y = 3x
    backpropagate(y)
    @test y == 6.0
    @test x.d == 3.0
    
    assign(x, 2.0)
    assign(y, 5.0)
    z = 3x + y
    backpropagate(z)
    @test z == 11.0
    @test x.d == 3.0
    @test y.d == 1.0

    t = 2z
    backpropagate(t)
    @test z == 11.0
    @test x.d == 6.0
    @test y.d == 2.0
end

end
