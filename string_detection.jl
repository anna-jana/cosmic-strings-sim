# string contention method from Moore at al., Axion dark matter: strings and their cores, Appendix A.2
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
        sum(sqrt(cyclic_dist_squared(p, s[i], s[mod1(i + 1, length(s))])) for i in 1:length(s))
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

const c = 0.41238

function compute_gamma_factors(s::AbstractState, i1, i2, i3, i4)
    sum_v2 = 0.0
    sum_v = 0.0
    sum_gamma = 0.0

    for i in (i1, i2, i3, i4)
        a = tau_to_a(s.tau)
        H = t_to_H(tau_to_t(s.tau))

        psi = s.psi[i]
        psi_dot = s.psi_dot[i]

        phi = psi / a
        phi_dot = psi_dot / a^2 - H * phi

        # Moore et al: Axion dark matter: strings and their cores, eq. A10
        gamma2_times_v2 = abs2(phi_dot) / c^2 * (1 + abs2(phi) / (8 * c^2)) + (phi' * phi_dot + phi * phi_dot')^2 / (16 * c^4)
        # gamma2_times_v2 = v^2 / (1 - v^2) = x
        # v^2 = (1 - v^2) * x
        # x = v^2 + v^2 x = v^2 (1 + x)
        # v^2 = x / (1 + x)
        # v = sqrt(x / (1 + x))
        v2 = gamma2_times_v2 / (1 + gamma2_times_v2)
        v = sqrt(v2)
        gamma = sqrt(1 / (1 - v^2))

        # weighted by the gamma factor i.e. the energy int gamma ds instead of the string length int ds
        sum_v += v * gamma
        sum_v2 += v2 * gamma
        sum_gamma += gamma^2
    end

    return sum_v, sum_v2, sum_gamma
end

function detect_strings(s :: SingleNodeState, p :: Parameter)
    string_points = Set{SVector{3, Float64}}()
    mean_v = 0.0
    mean_v2 = 0.0
    mean_gamma = 0.0

    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds for ix in 1:p.N
                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(mod1(ix + 1, p.N), iy, iz)
                i3 = CartesianIndex(mod1(ix + 1, p.N), mod1(iy + 1, p.N), iz)
                i4 = CartesianIndex(ix, mod1(iy + 1, p.N), iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    push!(string_points, SVector(ix - 1 + 0.5, iy - 1 + 0.5, iz))
                    mean_v, mean_v2, mean_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                end

                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(ix, mod1(iy + 1, p.N), iz)
                i3 = CartesianIndex(ix, mod1(iy + 1, p.N), mod1(iz + 1, p.N))
                i4 = CartesianIndex(ix, iy, mod1(iz + 1, p.N))
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    push!(string_points, SVector(ix, iy - 1 + 0.5, iz - 1 + 0.5))
                    mean_v, mean_v2, mean_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                end

                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(ix, iy, mod1(iz + 1, p.N))
                i3 = CartesianIndex(mod1(ix + 1, p.N), iy, mod1(iz + 1, p.N))
                i4 = CartesianIndex(mod1(ix + 1, p.N), iy, iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    push!(string_points, SVector(ix - 1 + 0.5, iy, iz - 1 + 0.5))
                    mean_v, mean_v2, mean_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                end
            end
        end
    end

    mean_v /= length(string_points)
    mean_v2 /= length(string_points)
    mean_gamma /= length(string_points)

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

    return strings, mean_v, mean_v2, mean_gamma
end

@inline dist(p1, p2) = norm(p1 - p2)

function check_if_at_boundary(s::MPIState, new::SVector{3, Float64})
    return abs(new[1]) <= sqrt(3) ||
           abs(new[2]) <= sqrt(3) ||
           abs(new[3]) <= sqrt(3) ||
           abs(new[1] - s.lnx) <= sqrt(3) ||
           abs(new[2] - s.lny) <= sqrt(3) ||
           abs(new[3] - s.lnz) <= sqrt(3)
end

function detect_strings(s::MPIState, p::Parameter)
    string_points = Set{SVector{3, Float64}}()
    points = SVector{3, Float64}[]
    string_length = 0.0
    sum_v = 0.0
    sum_v2 = 0.0
    sum_gamma = 0.0
    num_points = 0

    @inbounds for iz in 1:s.lnz
        @inbounds for iy in 1:s.lny
            @inbounds for ix in 1:s.lnx
                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(ix + 1, iy, iz)
                i3 = CartesianIndex(ix + 1, iy + 1, iz)
                i4 = CartesianIndex(ix, iy + 1, iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    new = SVector(ix - 1 + 0.5, iy - 1 + 0.5, iz)
                    if check_if_at_boundary(s, new)
                        string_length += 3/2
                    end
                    push!(string_points, new)
                    sum_v, sum_v2, sum_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                    num_points += 1
                end

                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(ix, iy + 1, iz)
                i3 = CartesianIndex(ix, iy + 1, iz + 1)
                i4 = CartesianIndex(ix, iy, iz + 1)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    new = SVector(ix, iy - 1 + 0.5, iz - 1 + 0.5)
                    if check_if_at_boundary(s, new)
                        string_length += 3/2
                    end
                    push!(string_points, new)
                    sum_v, sum_v2, sum_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                    num_points += 1
                end

                i1 = CartesianIndex(ix, iy, iz)
                i2 = CartesianIndex(ix, iy, iz + 1)
                i3 = CartesianIndex(ix + 1, iy, iz + 1)
                i4 = CartesianIndex(ix + 1, iy, iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4])
                    new = SVector(ix - 1 + 0.5, iy, iz - 1 + 0.5)
                    if check_if_at_boundary(s, new)
                        string_length += 3/2
                    end
                    push!(string_points, new)
                    sum_v, sum_v2, sum_gamma += compute_gamma_factors(s, i1, i2, i3, i4)
                    num_points += 1
                end
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

    num_points = MPI.Reduce(num_points, +, s.root, s.comm)

    sum_v = MPI.Reduce(sum_v, +, s.root, s.comm)
    sum_v2 = MPI.Reduce(sum_v2, +, s.root, s.comm)
    sum_gamma = MPI.Reduce(sum_gamma, +, s.root, s.comm)

    if s.rank == s.root
        mean_v = sum_v / num_points
        mean_v2 = sum_v2 / num_points
        mean_gamma = sum_gamma / num_points
    else
        mean_v = mean_v2 = mean_gamma = 0.0
    end

    string_length = MPI.Reduce(string_length, +, s.root, s.comm)

    return total_string_length(s, p, string_length), points, mean_v, mean_v2, mean_gamma

end


