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
end

end

BasicTest.test()
