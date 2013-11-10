module SecondDerivativeTest

using ReverseAutoDiff
using Base.Test

function test()
    x = RAD2(3.0)
    y = x
    backpropagate(y)
    @test value(y) == 3.0
    @test partial(x) == 1.0
    @test partial2(x) == 0.0

    y = 2x
    backpropagate(y)
    @test value(y) == 6.0
    @test partial(x) == 2.0
    @test partial2(x) == 0.0

    x = RAD2(2.0)
    y = RAD2(5.0)
    z = 3x + y
    backpropagate(z)
    @test z == 11.0
    @test partial(x) == 3.0
    @test partial2(x) == 0.0
    @test partial(y) == 1.0
    @test partial2(y) == 0.0

    x = RAD2(3.0)
    y = x*x
    backpropagate(y)
    @test value(y) == 9.0
    @test_approx_eq partial(x) 6.0
    @test_approx_eq partial2(x) 2.0
end

end

SecondDerivativeTest.test()
