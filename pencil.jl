using MPI
using PencilFFTs
using PencilArrays
using Random
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

N = 64
dim = (N, N, N)

pen = Pencil(dim, comm)

if rank == 0
    println(decomposition(pen))
    println(topology(pen))
end

# doing an fft
transform = Transforms.RFFT()
plan = PencilFFTPlan(pen, transform)

if rank == 0
    println("created fft plan")
end

phi = allocate_input(plan)
Random.seed!(rank)
Random.randn!(phi)
phi_hat = allocate_output(plan)

if rank == 0
    println("allocated memory")
end

mul!(phi_hat, plan, phi)

if rank == 0
    println("executed fft")
end

local_sum = sum(parent(phi_hat))
s = MPI.Reduce(local_sum, +, 0, comm)
if rank == 0
    println(s)
end

# constructing a PencilArray from exisiting local array
# local_dim = size_local(pen, MemoryOrder())  # dimensions must be in memory order!
# data = zeros(local_dim)
# A = PencilArray(pencil, data)

# acessing array elements

MPI.Finalize()
exit(0)


