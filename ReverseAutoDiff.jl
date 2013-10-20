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

function +(x::Acc, y::Acc)
    Acc(x.v + y.v, 1.0, (x,y), (1,1))
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
    
    assign(x, 2.0)
    assign(y, 5.0)
    z = 3x + y
    backpropagate(z)
    @test z.v == 11.0
    @test x.d == 3.0
    @test y.d == 1.0
end

end
