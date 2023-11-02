# mpirun -n 4 julia --project=. run_mpi.jl

include("AxionStrings.jl")

p = AxionStrings.Parameter(2.0, 3.0, 1e-2, 42, 1.0, 20, 1)
s = AxionStrings.MPIState(p)

AxionStrings.finish_mpi!(s)


