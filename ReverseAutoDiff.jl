module ReverseAutoDiff

type RAD
    v
    parents
    partials
    d

end

RAD(x, parents, partials) = RAD(x, parents, partials, zero(x))
RAD(x) = RAD(x, (), ())

function *(a::RAD, b::RAD)
    RAD(a.v * b.v, (a,b), (b.v,a.v))
end

function restart_backpropagation(v::RAD)
    v.d = zero(v.v)
    for i in 1:length(v.parents)
        restart_backpropagation(v.parents[i])
    end
end

function backpropagate(v::RAD)
    restart_backpropagation(v)
    backpropagate(v, one(v.v))
end

function backpropagate(v::RAD, r)
    v.d += r
    for i in 1:length(v.parents)
        backpropagate(v.parents[i], r * v.partials[i])
    end
end

import Base.show
show(io::IO, x::RAD) = (println(io, "RAD"); show(io, x, 1))
function show(io::IO, x::RAD, indent_count, r = Nothing)
    indent = repeat("  ", indent_count)
    print(io, indent)
    if r != Nothing
        print(io, "r=")
        show(io, r)
        print(io, " ")
    end
    print(io, "id=")
    show(io, object_id(x))
    print(io, " v=")
    show(io, x.v)
    print(io, " d=")
    show(io, x.d)
    println(io)
    for i in 1:length(x.parents)
        show(io, x.parents[i], indent_count + 1, x.partials[i])
    end
end

function assign(a::RAD, x)
    a.d = zero(a.v)
    a.v = x
    a.parents = ()
    a.partials = ()
end

function *(x, y::RAD)
    RAD(y.v * x, (y,), (x,))
end

function .*(x::RAD, y)
    RAD(x.v .* y, (x,), (y,))
end

function +(x::RAD, y::RAD)
    RAD(x.v + y.v, (x,y), (1,1))
end

function ==(x::RAD, y)
    x.v == y
end

function -(x::RAD, y)
    RAD(x.v - y, (x,), (1,))
end

function -(x::RAD, y::RAD)
    RAD(x.v - y.v, (x,y), (1,-1))
end

import Base.tanh
function tanh(x::RAD)
    t = tanh(x.v)
    RAD(t, (x,), (1-t^2,))
end

using Base.Test

function test()
    x = RAD(3.0)
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
    @test t == 22.0
    @test z.d == 2.0
    @test x.d == 6.0
    @test y.d == 2.0

    assign(x, 0.1234)
    A = 1.7159
    S = 2 / 3
    y = A * tanh(S * x)
    backpropagate(y)
    @test x.d == (A*(1 - tanh(S*x.v)^2) * S)

    assign(x, 3.0)
    y = x*x
    backpropagate(y)
    @test y.v == 9.0
    @test x.d == 6.0

    a = [RAD(i) for i in 1.0:30.0]
    b = a[1]
    for i in 2:length(a)
        b *= a[i]
    end
    d = [b.v/a[i].v for i in 1:length(a)]
    
    backpropagate(b)
    for i in 1:length(a)
        @test_approx_eq a[i].d d[i]
    end

    x = RAD(1.23)
    y = RAD(2.34)
    z = ((x + y) * (x - 2.5)) * x + x
    backpropagate(z)
    @test_approx_eq x.d (x.v*(y.v+x.v)+(x.v-2.5)*(y.v+x.v)+(x.v-2.5)*x.v + 1)

    a = [RAD(x) for x in rand(10)]
    b = sum(a .* 2)
    backpropagate(b)
    @test_approx_eq a[2].d 2.0
end

end
