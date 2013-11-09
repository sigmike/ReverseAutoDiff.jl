
tests = [
  "basic",
]

for t in tests
    println(" * $(t)")
    include("$(t).jl")
end
