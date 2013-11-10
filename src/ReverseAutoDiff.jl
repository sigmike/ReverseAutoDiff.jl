module ReverseAutoDiff

export
    RAD,
    RAD2,
    backpropagate,
    value,
    partial,
    partial2

type Record
    variable
    partial
end

type Record2
    variable
    partial
    partial2
end

abstract RADN

type RAD{T} <: RADN
    value::T
    partial::T
    tape::Array{Record,1}
end

type RAD2{T} <: RADN
    value::T
    partial::T
    partial2::T
    tape::Array{Record2,1}
end

RAD(value) = RAD(value, zero(value), Record[])
RAD(value, tape::Array{Record,1}) = RAD(value, zero(value), tape)

RAD2(value) = RAD2(value, zero(value), zero(value), Record2[])
RAD2(value, tape::Array{Record2,1}) = RAD2(value, zero(value), zero(value), tape)

value(x::RAD) = x.value
partial(x::RAD) = x.partial

value(x::RAD2) = x.value
partial(x::RAD2) = x.partial
partial2(x::RAD2) = x.partial2

function restart_backpropagation(x::RAD)
    x.partial = zero(value(x))
    for record in x.tape
        restart_backpropagation(record.variable)
    end
end

function restart_backpropagation(x::RAD2)
    x.partial = zero(value(x))
    x.partial2 = zero(value(x))
    for record in x.tape
        restart_backpropagation(record.variable)
    end
end

function backpropagate(x::RAD)
    restart_backpropagation(x)
    backpropagate(x, one(value(x)))
end

function backpropagate(x::RAD2)
    restart_backpropagation(x)
    backpropagate(x, one(value(x)), zero(value(x)))
end

function backpropagate(x::RAD, partial)
    x.partial += partial
    for record in x.tape
        backpropagate(record.variable, partial * record.partial)
    end
end

function backpropagate(x::RAD2, partial, partial2)
    x.partial += partial
    x.partial2 += partial2
    for record in x.tape
        backpropagate(record.variable, partial * record.partial, record.partial2)
    end
end

==(x::RADN, y) = (value(x) == y)

*(x::Real, y::RAD) = RAD(x * value(y), [Record(y, x)])
*(x::RAD, y::Real) = RAD(value(x) * y, [Record(x, y)])
*(x::RAD, y::RAD) = RAD(value(x) * value(y), [Record(x, value(y)), Record(y, value(x))])
.*(x::RAD, y) = RAD(value(x) .* y, [Record(x, y)])

*(x::Real, y::RAD2) = RAD2(x * value(y), [Record2(y, x, 0)])
*(x::RAD2, y::RAD2) = RAD2(value(x) * value(y), [Record2(x, value(y), one(value(y))), Record2(y, value(x), one(value(x)))])

+(x::Real, y::RAD) = RAD(x+value(y), [Record(y, one(value(y)))])
+(x::RAD, y::Real) = RAD(value(x)+y, [Record(x, one(value(x)))])
+(x::RAD, y::RAD) = RAD(value(x) + value(y), [Record(x, one(value(x))), Record(y, one(value(y)))])
+(x::RAD2, y::RAD2) = RAD2(value(x) + value(y), [Record2(x, one(value(x)), zero(value(x))), Record2(y, one(value(y)), zero(value(y)))])

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
