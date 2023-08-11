module CosmicStrings

using PyPlot
using FFTW
using Random
using StaticArrays
using LinearAlgebra

Base.@kwdef struct Parameter
    # simulation domain in time in log units
    log_start :: Float64
    log_end :: Float64
    # comoving length of the simulation box in units of 1/m_r
    L :: Float64
    # number of grid points in one dimension
    N :: Int
    # time step size in tau
    Delta_tau :: Float64
    # seed for random number generator
    seed :: Int
    k_max :: Float64
    dx :: Float64
    tau_start :: Float64
    tau_end :: Float64
    tau_span :: Float64
    nsteps :: Int

    nbins :: Int
    radius :: Int

    # optaning analysis data from simulation
    compute_energy_interval_tau :: Float64
    compute_strings_interval_tau :: Float64
    compute_spectrum_interval_tau :: Float64
    compute_energy_interval :: Int
    compute_strings_interval :: Int
    compute_spectrum_interval :: Int
end

Base.@kwdef mutable struct State
    tau :: Float64
    step :: Int
    phi :: Array{Complex{Float64}, 3}
    phi_dot :: Array{Complex{Float64}, 3}
    phi_dot_dot :: Array{Complex{Float64}, 3}
    next_phi :: Array{Complex{Float64}, 3}
    next_phi_dot :: Array{Complex{Float64}, 3}
    next_phi_dot_dot :: Array{Complex{Float64}, 3}
end

log_to_H(l) = 1.0 / exp(l)
H_to_t(H) = 1 / (2*H)
t_to_H(t) = 1 / (2*t)
H_to_log(H) = log(1/H)
t_to_tau(t) = -2*sqrt(t)
log_to_tau(log) = t_to_tau(H_to_t(log_to_H(log)))
t_to_a(t) = sqrt(t)
tau_to_t(tau) = -0.5*(tau)^2
tau_to_a(tau) = -0.5*tau
tau_to_log(tau) = H_to_log(t_to_H(tau_to_t(tau)))

const field_max = 1 / sqrt(2)

function new_field_array(p :: Parameter)
    return Array{Float64, 3}(undef, (p.N, p.N, p.N))
end

function random_field(p :: Parameter)
    hat = new_field_array(p)
    ks = fftfreq(p.N, 1 / p.dx) .* (2*pi)
    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds @simd for ix in 1:p.N
                kx = ks[ix]
                ky = ks[iy]
                kz = ks[iz]
                k = sqrt(kx^2 + ky^2 + kz^2)
                hat[ix, iy, iz] = k <= p.k_max ? (rand()*2 - 1) * field_max : 0.0
            end
        end
    end
    return ifft(hat)
end

function init(;
        log_start = 2.0,
        log_end = 3.0,
        Delta_tau = -1e-2,
        seed = 42,
        k_max = 1.0,
        nbins = 20,
        radius = 1,
        compute_energy_interval_tau = 0.1,
        compute_strings_interval_tau = 0.1,
        compute_spectrum_interval_tau = 1.0,
    )
    Random.seed!(seed)

    L = 1 / log_to_H(log_end)
    N = ceil(Int, L * tau_to_a(log_to_tau(log_end)))
    tau_start = log_to_tau(log_start)
    tau_end = log_to_tau(log_end)
    tau_span = tau_end - tau_start

    p = Parameter(
        log_start=log_start,
        log_end=log_end,
        L=L,
        N=N,
        Delta_tau=Delta_tau,
        seed=seed,
        k_max=k_max,
        dx=L / N, # L/N not L/(N-1) bc we have cyclic boundary conditions*...*...* N = 2
        tau_start=tau_start,
        tau_end=tau_end,
        tau_span=tau_span,
        nsteps=ceil(Int, tau_span / Delta_tau),
        nbins=nbins,
        radius=radius,
        compute_energy_interval_tau=compute_energy_interval_tau,
        compute_strings_interval_tau=compute_strings_interval_tau,
        compute_spectrum_interval_tau=compute_spectrum_interval_tau,
        compute_energy_interval=floor(Int, Delta_tau / compute_energy_interval_tau),
        compute_strings_interval=floor(Int, Delta_tau / compute_strings_interval_tau),
        compute_spectrum_interval=floor(Int, Delta_tau / compute_spectrum_interval_tau),
   )

   s = State(
        tau=tau_start,
        step=0,
        phi=random_field(p),
        phi_dot=random_field(p),
        phi_dot_dot=new_field_array(p),
        next_phi=new_field_array(p),
        next_phi_dot=new_field_array(p),
        next_phi_dot_dot=new_field_array(p),
   )

   compute_force!(s.phi_dot_dot, s, p)

   return s, p
end

function compute_force!(out :: Array{Complex{Float64}, 3}, s :: State, p :: Parameter)
    scale_factor = tau_to_a(s.tau)
    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds @simd for ix in 1:p.N
                pot_force = s.phi[ix, iy, iz] * (abs2(s.phi[ix, iy, iz]) - 0.5*scale_factor)
                laplace = (- 6 * s.phi[ix, iy, iz] +
                    s.phi[mod1(ix + 1, p.N), iy, iz] +
                    s.phi[mod1(ix - 1, p.N), iy, iz] +
                    s.phi[ix, mod1(iy + 1, p.N), iz] +
                    s.phi[ix, mod1(iy - 1, p.N), iz] +
                    s.phi[ix, iy, mod1(iz + 1, p.N)] +
                    s.phi[ix, iy, mod1(iz - 1, p.N)]) / p.dx^2
                out[ix, iy, iz] = + laplace - pot_force
            end
        end
    end
end

function make_step!(s :: State, p :: Parameter)
    # propagate PDE using velocity verlet algorithm
    s.tau = p.tau_start + (s.step + 1) * p.Delta_tau

    # update the field ("position")
    @. s.next_phi = s.phi + p.Delta_tau*s.phi_dot + 0.5*p.Delta_tau^2*s.phi_dot_dot

    # update the field derivative ("velocity")
    compute_force!(s.next_phi_dot_dot, s, p)

    @. s.next_phi_dot = s.phi_dot + p.Delta_tau*0.5*(s.phi_dot_dot + s.next_phi_dot_dot)

    # swap current and next arrays
    (s.phi, s.phi_dot, s.phi_dot_dot, s.next_phi, s.next_phi_dot, s.next_phi_dot_dot) = (
       s.next_phi, s.next_phi_dot, s.next_phi_dot_dot, s.phi, s.phi_dot, s.phi_dot_dot)
end

function compute_energy(s :: State, p :: Parameter)
    a = tau_to_a(s.tau)
    theta = angle.(s.phi)
    radial = @. sqrt(2) * abs(s.phi) / a - 1

    mean_axion_kinetic = 0.0
    mean_axion_gradient = 0.0
    mean_radial_kinetic = 0.0
    mean_radial_gradient = 0.0
    mean_radial_potential = 0.0
    mean_interaction = 0.0

    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds @simd for ix in 1:p.N
                R = real(s.phi[ix, iy, iz])
                I = imag(s.phi[ix, iy, iz])
                R_dot = real(s.phi_dot[ix, iy, iz])
                I_dot = imag(s.phi_dot[ix, iy, iz])

                # axion
                # kinetic
                d_theta_d_tau = (I_dot * R - I * R_dot) / (R^2 - I^2)
                axion_kinetic = 0.5 / a^2 * d_theta_d_tau^2
                # gradient
                diff_theta_x = theta[mod1(ix + 1, p.N), iy, iz] - theta[mod1(ix - 1, p.N), iy, iz]
                diff_theta_y = theta[ix, mod1(iy + 1, p.N), iz] - theta[ix, mod1(iy - 1, p.N), iz]
                diff_theta_z = theta[ix, iy, mod1(iz - 1, p.N)] - theta[ix, iy, mod1(iz - 1, p.N)]
                axion_gradient = 0.5 / p.dx^2 * (diff_theta_x^2 + diff_theta_y^2 + diff_theta_z^2)

                # radial mode
                # kinetic
                d_r_d_tau = (R * R_dot + I * I_dot) / abs(s.phi[ix, iy, iz])
                radial_kinetic = 0.5 / a^2 * d_r_d_tau^2
                # gradient
                diff_radial_x = radial[mod1(ix + 1, p.N), iy, iz] - radial[mod1(ix - 1, p.N), iy, iz]
                diff_radial_y = radial[ix, mod1(iy + 1, p.N), iz] - radial[ix, mod1(iy - 1, p.N), iz]
                diff_radial_z = radial[ix, iy, mod1(iz - 1, p.N)] - radial[ix, iy, mod1(iz - 1, p.N)]
                radial_gradient = 0.5 / p.dx^2 * (diff_radial_x^2 + diff_radial_y^2 + diff_radial_z^2)

                # potential
                inner = radial[ix, iy, iz]^2 - 2.0*radial[ix, iy, iz]
                radial_potential = inner^2 / 8.0

                # interaction
                interaction = inner * (axion_kinetic + axion_gradient)

                mean_axion_kinetic += axion_kinetic
                mean_axion_gradient += axion_gradient
                mean_radial_kinetic += radial_kinetic
                mean_radial_gradient += radial_gradient
                mean_radial_potential += radial_potential
                mean_interaction += interaction
            end
        end
    end

    mean_axion_kinetic /= p.N^3
    mean_axion_gradient /= p.N^3
    mean_radial_kinetic /= p.N^3
    mean_radial_gradient /= p.N^3
    mean_radial_potential /= p.N^3
    mean_interaction /= p.N^3

    mean_axion_total = mean_axion_kinetic + mean_axion_gradient
    mean_radial_total = mean_radial_kinetic + mean_radial_gradient + mean_radial_potential
    mean_total = mean_axion_total + mean_radial_total + mean_interaction

    return (mean_axion_kinetic, mean_axion_gradient, mean_axion_total,
            mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_radial_total,
            mean_interaction, mean_total)
end

# (string contention method from Moore at al.)
@inline function crosses_real_axis(phi1 :: Complex{Float64}, phi2 :: Complex{Float64}) :: Bool
    return imag(phi1) * imag(phi2) < 0
end

@inline function handedness(phi1 :: Complex{Float64}, phi2 :: Complex{Float64}) :: Int
    return sign(imag(phi1 * conj(phi2)))
end

@inline function loop_contains_string(phi1 :: Complex{Float64}, phi2 :: Complex{Float64},
                              phi3 :: Complex{Float64}, phi4 :: Complex{Float64})
    loop = (
          crosses_real_axis(phi1, phi2) * handedness(phi1, phi2)
        + crosses_real_axis(phi2, phi3) * handedness(phi2, phi3)
        + crosses_real_axis(phi3, phi4) * handedness(phi3, phi4)
        + crosses_real_axis(phi4, phi1) * handedness(phi4, phi1)
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
                if loop_contains_string(s.phi[ix, iy, iz], s.phi[mod1(ix + 1, p.N), iy, iz],
                                        s.phi[mod1(ix + 1, p.N), mod1(iy + 1, p.N), iz], s.phi[ix, mod1(iy + 1, p.N), iz])
                    push!(string_points, SVector(ix - 1 + 0.5, iy - 1 + 0.5, iz))
                end
                if loop_contains_string(s.phi[ix, iy, iz], s.phi[ix, mod1(iy + 1, p.N), iz],
                                        s.phi[ix, mod1(iy + 1, p.N), mod1(iz + 1, p.N)], s.phi[ix, iy, mod1(iz + 1, p.N)])
                    push!(string_points, SVector(ix, iy - 1 + 0.5, iz - 1 + 0.5))
                end
                if loop_contains_string(s.phi[ix, iy, iz], s.phi[ix, iy, mod1(iz + 1, p.N)],
                                        s.phi[mod1(ix + 1, p.N), iy, mod1(iz + 1, p.N)], s.phi[mod1(ix + 1, p.N), iy, iz])
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

function plot_strings(params :: Parameter, strings :: Vector{Vector{SVector{3, Float64}}})
    fig = figure()
    fig.add_subplot(projection="3d")

    for string in strings
        xs = [string[1][1]]
        ys = [string[1][2]]
        zs = [string[1][3]]
        prev = string[1]
        color = nothing

        for p in string[2:end]
            if norm(p .- prev) <= sqrt(3)
                push!(xs, p[1])
                push!(ys, p[2])
                push!(zs, p[3])
            else
                l, = plot(xs, ys, zs, color=color)
                color = l.get_color()
                xs = [p[1]]
                ys = [p[2]]
                zs = [p[3]]
            end
            prev = p
        end

        if norm(string[1] - string[end]) <= sqrt(3)
            push!(xs, string[1][1])
            push!(ys, string[1][2])
            push!(zs, string[1][3])
        end

        plot(xs, ys, zs, color=color)
    end

    xlabel("x")
    ylabel("y")
    zlabel("z")

    return nothing
end

@inline function compute_theta_dot(a :: Float64, phi :: Complex{Float64}, phi_dot :: Complex{Float64})
    R = real(phi)
    I = imag(phi)
    R_dot = real(phi_dot)
    I_dot = imag(phi_dot)
    d_theta_d_tau = (I_dot * R - I * R_dot) / (R^2 - I^2)
    return d_theta_d_tau / a
end

@inline function calc_k_max_grid(n, d)
    if n % 2 == 0
        return 2 * pi * (n / 2) / (d*n)
    else
        return 2 * pi * ((n - 1) / 2) / (d*n)
    end
end

@inline function idx_to_k(p, idx)
    return idx < div(p.N, 2) ? idx : -div(p.N, 2) + idx  - div(p.N, 2)
end

@inline function substract_wave_numbers(p, i, j)
    k = idx_to_k(p, i)
    k_prime = idx_to_k(p, j)
    k_diff = mod(k - k_prime + div(p.N, 2), p.N) - div(p.N, 2)
    idx_diff = k_diff < 0 ? k_diff + p.N : k_diff
    return idx_diff
end


# compute PPSE (pseudo-power-spectrum-estimator) of the theta-dot field
# -> the spectrum of number denseties of axtion
function compute_spectrum(p :: Parameter, s :: State, strings :: Vector{Vector{SVector{3, Float64}}})
    a = tau_to_a(s.tau)
    theta_dot = compute_theta_dot.(a, s.phi, s.phi_dot)

    # compute W (string mask)
    W = fill(1.0 + 0.0im, (p.N, p.N, p.N))
    for string in strings
        for point in string
            @inbounds for x_offset in -p.radius:p.radius
                @inbounds for y_offset in -p.radius:p.radius
                    @inbounds @simd for z_offset in -p.radius:p.radius
                        r2 = x_offset^2 + y_offset^2 + z_offset^2
                        if r2 <= p.radius^2
                            ix = mod1(floor(Int, point[1] + 1.0), p.N)
                            iy = mod1(floor(Int, point[2] + 1.0), p.N)
                            iz = mod1(floor(Int, point[3] + 1.0), p.N)
                            W[ix, iy, iz] = zero(eltype(W))
                        end
                    end
                end
            end
        end
    end

    # mask out the strings in theta dot
    theta_dot .*= W

    # prepare histogram
    dx_physical = p.dx * a
    kmax = calc_k_max_grid(p.N, dx_physical)

    Delta_k = 2*pi / (p.N * dx_physical)
    bin_width = kmax / p.nbins

    surface_element = map(1:p.nbins) do i
        vol = 4.0/3.0 * pi * (((i + 1)*bin_width)^3 - (i*bin_width)^3)
        area = 4*pi * (i*bin_width + bin_width/2)^2
        area / vol * Delta_k^3
    end

    physical_ks = fftfreq(p.N, 1 / dx_physical) * 2*pi

    # TODO: use preplaned ffts
    theta_dot_fft = fft(theta_dot)

    spheres = [Tuple{Int, Int, Int}[] for i in 1:p.nbins]
    for i in 1:p.nbins
        bin_k_min = i * bin_width
        bin_k_max = bin_k_min + bin_width
        @inbounds for iz in 1:p.N
            @inbounds for iy in 1:p.N
                @inbounds for ix in 1:p.N
                    k2 = physical_ks[ix]^2 + physical_ks[iy]^2 + physical_ks[iz]^2
                    if k2 >= bin_k_min^2 && k2 <= bin_k_max^2
                        push!(spheres[i], (ix, iy, iz))
                    end
                end
            end
        end
    end

    # P_field(k) = k^2 / L^3 \int d \Omega / 4\pi 0.5 * | field(k) |^2
    spectrum_uncorrected = zeros(p.nbins)
    for i in 1:p.nbins
        bin_k = i * bin_width + bin_width/2.0
        for (ix, iy, iz) in spheres[i]
            spectrum_uncorrected[i] += abs2(theta_dot_fft[ix, iy, iz])
        end
        spectrum_uncorrected[i] *= surface_element[i]
        spectrum_uncorrected[i] *= bin_k^2 / p.L^3 / (4 * pi) * 0.5
    end

    W_fft = fft(W)

    # compute M
    # M = 1 / (L^3)^2 * \int d \Omega / 4\pi d \Omega' / 4\pi |W(\vec{k} - \vec{k}')|^2
    # NOTE: this is the computationally most expensive part!
    M = zeros(p.nbins, p.nbins)
    f = p.L^6 * (4 * pi)^2
    for i in 1:p.nbins
        for j in 1:p.nbins
            @show i, j
            # integrate spheres
            s_atomic = Threads.Atomic{Float64}(0.0)
            Threads.@threads for idx1 in spheres[i]
                for idx2 in spheres[j]
                    ix = substract_wave_numbers(p, idx1[1] - 1, idx2[1] - 1)
                    iy = substract_wave_numbers(p, idx1[2] - 1, idx2[2] - 1)
                    iz = substract_wave_numbers(p, idx1[3] - 1, idx2[3] - 1)
                    Threads.atomic_add!(s_atomic, abs2(@inbounds W_fft[ix + 1, iy + 1, iz + 1]))
                end
            end
            s = s_atomic[]
            s *= surface_element[i] * surface_element[j] / f
            M[i, j] = M[j, i] = s
        end
    end

    M_inv = inv(M)

    for i in 1:p.nbins
        for j in 1:p.nbins
            bin_k_1 = i * bin_width + bin_width/2;
            bin_k_2 = j * bin_width + bin_width/2;
            M_inv[i, j] = M_inv[i, j] * (2 * pi)^2 / (bin_width * bin_k_1^2 * bin_k_2^2)
        end
    end

    spectrum_corrected = M_inv * spectrum_uncorrected # NOTE: matrix multiply!

    for i in p.nbins
        bin_k = i * bin_width + bin_width/2
        spectrum_corrected[i] *= bin_k^2 / p.L^3 / (2*pi^2) * bin_width
    end

    return physical_ks, spectrum_corrected
end


function run_simulation!(s :: State, p :: Parameter)
    energies = []
    strings = []
    spectra = []
    for i in p.nsteps
        make_step!(s, p)
        if i % p.compute_energy_interval == 0
            push!(energies, (s.tau, compute_energy(s, p)))
        end
        if i % p.compute_strings_interval == 0 || i % p.compute_spectrum_interval == 0
            strings = detect_strings(s, p)
        end
        if i % p.compute_strings_interval == 0
            push!(strings, (tau, map(length, strings)))
        end
        if i % p.compute_spectrum_interval == 0
            push!(spectra, compute_spectrum(s, p, strings))
        end
    end
    return energies, strings, spectra
end

end

sim, params = CosmicStrings.init()
