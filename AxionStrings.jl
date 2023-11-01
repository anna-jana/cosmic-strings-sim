module AxionStrings

using FFTW
using Random
using StaticArrays
using LinearAlgebra
using PyPlot
using DelimitedFiles
using Interpolations
using Statistics
using MPI
using PencilArrays
using PencilFFTs
using AbstractFFTs

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
    # params for spectrum computation
    nbins :: Int
    radius :: Int
end

Base.@kwdef mutable struct State
    tau :: Float64
    step :: Int
    psi :: Array{Complex{Float64}, 3}
    psi_dot :: Array{Complex{Float64}, 3}
    psi_dot_dot :: Array{Complex{Float64}, 3}
    next_psi_dot_dot :: Array{Complex{Float64}, 3}
end

log_to_H(l) = 1.0 / exp(l)
H_to_t(H) = 1 / (2*H)
t_to_H(t) = 1 / (2*t)
H_to_log(H) = log(1/H)
t_to_tau(t) = 2*sqrt(t)
log_to_tau(log) = t_to_tau(H_to_t(log_to_H(log)))
t_to_a(t) = sqrt(t)
tau_to_t(tau) = 0.5*(tau)^2
tau_to_a(tau) = 0.5*tau
tau_to_log(tau) = H_to_log(t_to_H(tau_to_t(tau)))

const field_max = 1 / sqrt(2)

function sim_params_from_physical_scale(log_end)
    L = 1 / log_to_H(log_end)
    N = ceil(Int, L * tau_to_a(log_to_tau(log_end)))
    return L, N
end

function init_parameter(;
        log_start,
        log_end,
        Delta_tau,
        seed,
        k_max,
        nbins,
        radius,
    )
    Random.seed!(seed)

    L, N = sim_params_from_physical_scale(log_end)

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
   )

    return p
end

include("propagation.jl")
include("single_node.jl")
include("mpi.jl")
include("energy.jl")
include("strings.jl")
include("spectrum.jl")

end
