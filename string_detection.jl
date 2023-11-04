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

function total_string_length(s::AbstractState, p::Parameter, l::Float64)
    a = tau_to_a(s.tau)
    t = tau_to_t(s.tau)
    return a * l / (p.L^3 * a^3) * t^2
end

function total_string_length(s::AbstractState, p::Parameter, strings::Vector{Vector{SVector{3, Float64}}})
    l = p.dx * sum(strings) do s
        sum(cyclic_dist_squared(p, s[i], s[mod1(i + 1, length(s))])^2 for i in 1:length(s))
    end
    return total_string_length(s, p, l)
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

function detect_strings(s :: SingleNodeState, p :: Parameter)
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

@inline function dist(p1, p2)
    return norm(p1 - p2)
end

function detect_strings(s::MPIState, p::Parameter)
    string_points = Set{SVector{3, Float64}}()
    points = SVector{3, Float64}[]
    string_length = 0.0

    @inbounds for iz in 1:s.lnz
        @inbounds for iy in 1:s.lny
            @inbounds for ix in 1:s.lnx
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[ix + 1, iy, iz],
                                        s.psi[ix + 1, iy + 1, iz], s.psi[ix, iy + 1, iz])
                    new = SVector(ix - 1 + 0.5, iy - 1 + 0.5, iz)
                end
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[ix, iy + 1, iz],
                                        s.psi[ix, iy + 1, iz + 1], s.psi[ix, iy, iz + 1])
                    new = SVector(ix, iy - 1 + 0.5, iz - 1 + 0.5)
                end
                if loop_contains_string(s.psi[ix, iy, iz], s.psi[ix, iy, iz + 1],
                                        s.psi[ix + 1, iy, iz + 1], s.psi[ix + 1, iy, iz])
                    new = SVector(ix - 1 + 0.5, iy, iz - 1 + 0.5)
                end
                if abs(new[1]) <= sqrt(3) ||
                   abs(new[2]) <= sqrt(3) ||
                   abs(new[3]) <= sqrt(3) ||
                   abs(new[1] - s.lnx) <= sqrt(3) ||
                   abs(new[2] - s.lny) <= sqrt(3) ||
                   abs(new[3] - s.lnz) <= sqrt(3)
                   string_length += 3/2
                end
                push!(string_points, new)
            end
        end
    end

    while !isempty(string_points)
        first_point = pop!(string_points)
        push!(points, first_point)
        last_point = first_point
        current_length = 1

        while true
            if isempty(string_points)
                # in mpi we leaf out the warning
                # if cyclic_dist_squared(p, last_point, first_point) >= sqrt(3)
                #     @warn "no points left but string isnt closed"
                # end
                break
            end

            closest = argmin(point -> dist(last_point, point), string_points)
            dist_to_new = dist(last_point, closest)
            dist_to_first = dist(last_point, first_point)

            if current_length <= 2 || dist_to_new < dist_to_first
                delete!(string_points, closest)
                push!(points, closest)
                last_point = closest
                string_length += dist_to_new
            else
                break # we closed the string
                string_length += dist_to_first
            end

        end

    end

    string_length = MPI.Reduce(string_length, +, s.root, s.comm)

    return total_string_length(s, p, string_length), points
end


