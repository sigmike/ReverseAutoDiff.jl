module ReverseAutoDiff

export
    rad,
    backpropagate,
    value,
    partial

type Record
    variable
    partial
end

type RAD
    value
    partial
    tape::Array{Record,1}
end

rad(value) = RAD(value, zero(value), Record[])
rad(value, tape::Array{Record,1}) = RAD(value, zero(value), tape)
value(x::RAD) = x.value
partial(x::RAD) = x.partial

function restart_backpropagation(x::RAD)
    x.partial = zero(value(x))
    for record in x.tape
        restart_backpropagation(record.variable)
    end
end

function backpropagate(x::RAD)
    restart_backpropagation(x)
    backpropagate(x, one(value(x)))
end

function backpropagate(x::RAD, partial)
    x.partial = partial
    for record in x.tape
        backpropagate(record.variable, partial * record.partial)
    end
end

*(x::Real, y::RAD) = rad(x * value(y), [Record(y, x)])

end # module
