module NeuralNetworkTest

using ReverseAutoDiff
using Base.Test

function test()
    # https://en.wikipedia.org/w/index.php?title=Backpropagation&oldid=577461892#Derivation
    n = 10
    w = [RAD(x) for x in rand(n)]
    x = rand(n)
    t = rand()
    activation(z) = 1.0 / (1 + exp(-z))
    y = activation(sum([w[i] * x[i] for i in 1:n]))
    E = 0.5 * ((t - y) ^ 2)
    backpropagate(E)
    for i in 1:n
        @test_approx_eq partial(w[i]) ((value(y) - t) * value(y) * (1 - value(y)) * x[i])
    end
end

end

NeuralNetworkTest.test()

