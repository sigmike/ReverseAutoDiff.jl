module BasicTest

using ReverseAutoDiff
using Base.Test

function test()
    x = rad(3.0)
    y = x
    backpropagate(y)
    @test value(y) == 3.0
    @test partial(x) == 1.0

    y = 2x
    backpropagate(y)
    @test value(y) == 6.0
    @test partial(x) == 2.0

    x = rad(2.0)
    y = rad(5.0)
    z = 3x + y
    backpropagate(z)
    @test z == 11.0
    @test partial(x) == 3.0
    @test partial(y) == 1.0
end

end

BasicTest.test()
