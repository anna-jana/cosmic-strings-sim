module AxionStrings

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

include("energy.jl")
include("strings.jl")
include("spectrum.jl")

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
