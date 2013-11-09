
tests = [
  "basic",
  "neural_network",
]

for t in tests
    println(" * $(t)")
    include("$(t).jl")
end
