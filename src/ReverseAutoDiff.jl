module ReverseAutoDiff

export
    RAD,
    backpropagate,
    value,
    partial

type Record
    variable
    partial
end

type RAD{T}
    value::T
    partial::T
    tape::Array{Record,1}
end

RAD(value) = RAD(value, zero(value), Record[])
RAD(value, tape::Array{Record,1}) = RAD(value, zero(value), tape)

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
    x.partial += partial
    for record in x.tape
        backpropagate(record.variable, partial * record.partial)
    end
end

==(x::RAD, y) = (value(x) == y)

*(x::Real, y::RAD) = RAD(x * value(y), [Record(y, x)])
*(x::RAD, y::Real) = RAD(value(x) * y, [Record(x, y)])
*(x::RAD, y::RAD) = RAD(value(x) * value(y), [Record(x, value(y)), Record(y, value(x))])
.*(x::RAD, y) = RAD(value(x) .* y, [Record(x, y)])

+(x::Real, y::RAD) = RAD(x+value(y), [Record(y, one(value(y)))])
+(x::RAD, y::Real) = RAD(value(x)+y, [Record(x, one(value(x)))])
+(x::RAD, y::RAD) = RAD(value(x) + value(y), [Record(x, one(value(x))), Record(y, one(value(y)))])

-(x::Real, y::RAD) = RAD(x-value(y), [Record(y, -one(value(y)))])
-(x::RAD, y::Real) = RAD(value(x)-y, [Record(x, one(value(x)))])
-(x::RAD, y::RAD) = RAD(value(x) - value(y), [Record(x, one(value(x))), Record(y, -one(value(y)))])
-(x::RAD) = RAD(-value(x), [Record(x, -one(value(x)))])

/(x::Real, y::RAD) = RAD(x/value(y), [Record(y, -x/(value(y)^2))])
/(x::RAD, y::RAD) = RAD(value(x)/value(y), [Record(x, one(value(x))/value(y)), Record(y, -value(x)/(value(y)^2))])

import Base.abs
function abs(x::RAD)
    if value(x) == zero(value(x))
        throw(DomainError())
    end
    if value(x) < zero(value(x))
        -x
    else
        x
    end
end

import Base.exp
function exp(x::RAD)
    exp_x = exp(value(x))
    RAD(exp_x, [Record(x, exp_x)])
end

import Base.tanh
function tanh(x::RAD)
    t = tanh(value(x))
    RAD(t, [Record(x, one(t)-t^2)])
end

end # module
