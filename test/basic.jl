module BasicTest

using ReverseAutoDiff
using Base.Test

function test()
    x = RAD(3.0)
    y = x
    backpropagate(y)
    @test_approx_eq value(y) 3.0
    @test_approx_eq partial(x) 1.0

    y = 2x
    backpropagate(y)
    @test value(y) == 6.0
    @test partial(x) == 2.0

    x = RAD(2.0)
    y = RAD(5.0)
    z = 3x + y
    backpropagate(z)
    @test z == 11.0
    @test partial(x) == 3.0
    @test partial(y) == 1.0

    t = 2z
    backpropagate(t)
    @test t == 22.0
    @test partial(z) == 2.0
    @test partial(x) == 6.0
    @test partial(y) == 2.0

    x = RAD(0.1234)
    A = 1.7159
    S = 2 / 3
    y = A * tanh(S * x)
    backpropagate(y)
    @test partial(x) == (A*(1 - tanh(S*value(x))^2) * S)

    x = RAD(3.0)
    y = x*x
    backpropagate(y)
    @test value(y) == 9.0
    @test partial(x) == 6.0

    a = [RAD(i) for i in 1.0:30.0]
    b = a[1]
    for i in 2:length(a)
        b *= a[i]
    end
    d = [value(b)/value(a[i]) for i in 1:length(a)]

    backpropagate(b)
    for i in 1:length(a)
        @test_approx_eq partial(a[i]) d[i]
    end

    x = RAD(1.23)
    y = RAD(2.34)
    z = ((x + y) * (x - 2.5)) * x + x
    backpropagate(z)
    @test_approx_eq partial(x) (value(x)*(value(y)+value(x))+(value(x)-2.5)*(value(y)+value(x))+(value(x)-2.5)*value(x) + 1)

    a = [RAD(x) for x in rand(10)]
    b = sum(a .* 2)
    backpropagate(b)
    @test_approx_eq partial(a[2]) 2.0
end

end

BasicTest.test()
