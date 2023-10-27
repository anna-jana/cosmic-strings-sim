# (string contention method from Moore at al.)
@inline function crosses_real_axis(psi1 :: Complex{Float64}, psi2 :: Complex{Float64}) :: Bool
    return imag(psi1) * imag(psi2) < 0
end

@inline function handedness(psi1 :: Complex{Float64}, psi2 :: Complex{Float64}) :: Int
    return sign(imag(psi1 * conj(psi2)))
end

@inline function loop_contains_string(psi1 :: Complex{Float64}, psi2 :: Complex{Float64},
                              psi3 :: Complex{Float64}, psi4 :: Complex{Float64})
    loop = (
          crosses_real_axis(psi1, psi2) * handedness(psi1, psi2)
        + crosses_real_axis(psi2, psi3) * handedness(psi2, psi3)
        + crosses_real_axis(psi3, psi4) * handedness(psi3, psi4)
        + crosses_real_axis(psi4, psi1) * handedness(psi4, psi1)
    )
    return abs(loop) == 2
end

@inline function cyclic_dist_squared_1d(p :: Parameter, x1 :: Float64, x2 :: Float64) :: Float64
    return min((x1 - x2)^2, (p.N - x1 + x2)^2, (p.N - x2 + x1)^2)
end

@inline function cyclic_dist_squared(p :: Parameter, p1 :: SVector{3, Float64}, p2 :: SVector{3, Float64}) :: Float64
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (
        cyclic_dist_squared_1d(p, x1, x2) +
        cyclic_dist_squared_1d(p, y1, y2) +
        cyclic_dist_squared_1d(p, z1, z2)
    )
end

# TODO; reverse loop order

function detect_strings(s :: State, p :: Parameter)
    string_points = Set{SVector{3, Float64}}()
    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds for ix in 1:p.N
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[mod1(ix + 1, p.N), iy, iz],
                                        s.psi[mod1(ix + 1, p.N), mod1(iy + 1, p.N), iz], s.psi[ix, mod1(iy + 1, p.N), iz])
                    push!(string_points, SVector(ix - 1 + 0.5, iy - 1 + 0.5, iz))
                end
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[ix, mod1(iy + 1, p.N), iz],
                                        s.psi[ix, mod1(iy + 1, p.N), mod1(iz + 1, p.N)], s.psi[ix, iy, mod1(iz + 1, p.N)])
                    push!(string_points, SVector(ix, iy - 1 + 0.5, iz - 1 + 0.5))
                end
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[ix, iy, mod1(iz + 1, p.N)],
                                        s.psi[mod1(ix + 1, p.N), iy, mod1(iz + 1, p.N)], s.psi[mod1(ix + 1, p.N), iy, iz])
                    push!(string_points, SVector(ix - 1 + 0.5, iy, iz - 1 + 0.5))
                end
            end
        end
    end

    strings = Vector{SVector{3, Float64}}[]

    while !isempty(string_points)
        current_string = [pop!(string_points)]

        while true
            if isempty(string_points)
                if cyclic_dist_squared(p, current_string[end], current_string[1]) >= sqrt(3)
                    @warn "no points left but string isnt closed"
                end
                break
            end

            closest = argmin(point -> cyclic_dist_squared(p, current_string[end], point), string_points)

            if length(current_string) <= 2 ||
               cyclic_dist_squared(p, current_string[end], closest) <
               cyclic_dist_squared(p, current_string[end], current_string[1])
                delete!(string_points, closest)
                push!(current_string, closest)
            else
                break # we closed the string
            end

        end

        push!(strings, current_string)
    end

    return strings
end

function total_string_length(s::State, p::Parameter, strings::Vector{Vector{SVector{3, Float64}}})
    a = tau_to_a(s.tau)
    t = tau_to_t(s.tau)
    l = p.dx * sum(strings) do s
        sum(cyclic_dist_squared(p, s[i], s[mod1(i + 1, length(s))]) for i in 1:length(s))
    end
    return a * l / (p.L^3 * a^3) * t^2
end
