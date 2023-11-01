# mpirun -n 4 julia --project=. run_mpi.jl

include("AxionStrings.jl")

p = AxionStrings.init_parameter(
    log_start = 2.0,
    log_end = 3.0,
    Delta_tau = 1e-2,
    seed = 42,
    k_max = 1.0,
    nbins = 20,
    radius = 1
)

s = AxionStrings.init_state_mpi(p)
