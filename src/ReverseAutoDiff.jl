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
    partial::RAD{T}
    tape::Array{Record,1}
end

RAD(value) = RAD(value, zero(value), Record[])
RAD(value, tape::Array{Record,1}) = RAD(value, zero(value), tape)

RAD2(value) = RAD2(value, RAD(zero(value)), Record[])

value(x::RADN) = x.value
partial(x::RADN) = x.partial
partial2(x::RADN) = partial(x.partial)

import Base.zero
zero(x::RAD) = RAD(zero(value(x)))

import Base.one
one(x::RAD) = RAD(one(value(x)))

function restart_backpropagation(x::RADN)
    x.partial = zero(partial(x))
    for record in x.tape
        restart_backpropagation(record.variable)
    end
end

function backpropagate(x::RADN)
    restart_backpropagation(x)
    backpropagate(x, one(partial(x)))
end

function backpropagate(x::RADN, partial)
    x.partial += partial
    for record in x.tape
        backpropagate(record.variable, partial * record.partial)
    end
end

==(x::RAD, y) = (value(x) == y)

*{T<:RADN}(x::Real, y::T) = T(x * value(y), [Record(y, x)])
*(x::RADN, y::Real) = RAD(value(x) * y, [Record(x, y)])
*(x::RADN, y::RADN) = RAD(value(x) * value(y), [Record(x, value(y)), Record(y, value(x))])
.*(x::RADN, y) = RAD(value(x) .* y, [Record(x, y)])

+(x::Real, y::RADN) = RAD(x+value(y), [Record(y, one(value(y)))])
+(x::RADN, y::Real) = RAD(value(x)+y, [Record(x, one(value(x)))])
+(x::RADN, y::RADN) = RAD(value(x) + value(y), [Record(x, one(value(x))), Record(y, one(value(y)))])

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
