using MPI
using Random

# mpirun -n 4 julia --project=. mpi_gol.jl

# splitting the grid into subboxes for each node such that the required communication
# i.e. the sum of surfaces of each subbox is minimal
# i.e. each subbox is as close to a cube as possible
function primes_to(n)
    primes = collect(2:n)
    i = 1
    while i <= length(primes)
        p = primes[i]
        j = i + 1
        while j <= length(primes)
            if primes[j] % p == 0
                deleteat!(primes, j)
            else
                j += 1
            end
        i += 1
        end
    end
    return primes
end

function prime_factors(n)
    primes = primes_to(n)
    factors = Int[]
    while n != 1
        for p in primes
            if n % p == 0
                push!(factors, p)
                n = div(n, p)
                break
            end
        end
    end
    return factors
end

function most_equal_products(factors, n)
    products = ones(Int, n)
    factors = sort(factors)
    i = 1
    for f in factors
        products[i] *= f
        i = mod1(i + 1, n)
    end
    return products
end

function split_grid_points_into_boxes(num_grid_points, num_boxes)
    box_sizes = fill(div(num_grid_points, num_boxes), num_boxes)
    for i in 1:num_grid_points % num_boxes
        box_sizes[i] += 1
    end
    return box_sizes
end

function get_node_boxes_sizes(grid_dimension, num_nodes)
    box_axis_lengths = most_equal_products(prime_factors(num_nodes),
                                           length(grid_dimension))
    return [split_grid_points_into_boxes(grid_axis_length, box_axis_length)
            for (grid_axis_length, box_axis_length)
            in zip(grid_dimension, box_axis_lengths)]
end

function distributed_grid(grid_dimension, num_nodes)
    boxes = get_node_boxes_sizes(grid_dimension, num_nodes)
    axis_lengths = map(length, boxes)
    zip_rank(xs) = zip(0:length(xs)-1, xs)
    node_id_to_box_size = Dict(zip_rank(Iterators.product(boxes...)))
    node_id_to_box_index = Dict(zip_rank(Iterators.product(
                                 map(b -> 1:length(b), boxes)...)))
    box_index_to_node_id = Dict([box_index => node_id
                                 for (node_id, box_index) in node_id_to_box_index])
    return (; boxes, axis_lengths, node_id_to_box_size, node_id_to_box_index, box_index_to_node_id)
end

# mpi setup
MPI.Init()
comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

print("begin setup $rank\n")

# setup the grid (divide the simulation into subboxes for each node)
N = 64 # TODO: use the value from parameters
global_grid_dimension = (N, N, N)
grid = distributed_grid(global_grid_dimension, nprocs)

# subbox setup
lnx, lny, lnz = grid.node_id_to_box_size[rank]
own_subbox = Array{Float64}(undef, lnx + 2, lny + 2, lnz + 2)
Random.seed!(rank)
# TODO: real setup
own = rand(lnx, lny, lnz)
own_subbox[2:end-1, 2:end-1, 2:end-1] = own

# setup for exchanging data with neighboring subboxes/nodes
own_subbox_index = grid.node_id_to_box_index[rank]

struct Neighbor
    rank::Int
    offset::Tuple{Int, Int, Int}
    receive_buffer::Array{Float64, 3}
end

function get_send_index(off, ln)
    if off == -1
        return 2
    elseif off == 0
        return 2:ln
    elseif off == 1
        return ln
    else
        # TODO: extend to larger neighborhoods
        error("invalid offset")
    end
end

function get_receive_index(off, ln)
    if off == -1
        return 1
    elseif off == 0
        return 2:ln
    elseif off == 1
        return ln + 1
    else
        # TODO: extend to larger neighborhoods
        error("invalid offset")
    end
end


neighbors = Neighbor[]
for dim in 1:3
    for side in (1, -1)
        offset = [0, 0, 0]
        offset[dim] = side

        neighbor_index = mod1.(own_subbox_index .+ offset, grid.axis_lengths)
        neighbor_id = grid.box_index_to_node_id[tuple(neighbor_index...)]


        rcvbuf = Array{Float64}(undef,
                                length(get_receive_index(offset[1], lnx)),
                                length(get_receive_index(offset[2], lny)),
                                length(get_receive_index(offset[3], lnz)),)

        neighbor = Neighbor(neighbor_id, tuple(offset...), rcvbuf)

        push!(neighbors, neighbor)
    end
end

if rank == 0
    print(neighbors)
end


MPI.Barrier(comm)

requests = MPI.Request[]

print("starting simulation loop $rank\n")

# simulation loop
nsteps = 100 # TODO: use the value from parameters
for i in 1:nsteps
    if rank == 0
        print("$i of $nsteps\n")
    end
    print("exchange data on node $rank\n")
    # exchange data with neighboring nodes/subboxes
    for neighbor in neighbors
        offx, offy, offz = neighbor.offset
        # slice of own data to send to neighbor
        to_send = @view own_subbox[
            get_send_index(offx, lnx),
            get_send_index(offy, lny),
            get_send_index(offz, lnz),
        ]
        send = MPI.Isend(to_send, neighbor.rank, i, comm)
        push!(requests, send)

        # receive data from neighbor
        receive = MPI.Irecv!(neighbor.receive_buffer, neighbor.rank, i, comm)
        push!(requests, receive)
    end
    MPI.Waitall(requests)

    # assign received data to own subbox
    for neighbor in neighbors
        offx, offy, offz = neighbor.offset
        ix = get_receive_index(offx, lnx)
        iy = get_receive_index(offy, lny)
        iz = get_receive_index(offz, lnz)
        own_subbox[ix, iy, iz] = neighbor.receive_buffer
    end

    # reuse the rquests list for next time
    empty!(requests)

    print("local update on node $rank\n")

    # do local update
    # TODO: real update
    for iz in 2:lnz
        for iy in 2:lny
            for ix in 2:lnx
                own_subbox[ix, iy, iz] = (
                   own_subbox[ix, iy, iz] +
                   own_subbox[ix + 1, iy, iz] +
                   own_subbox[ix - 1, iy, iz] +
                   own_subbox[ix, iy + 1, iz] +
                   own_subbox[ix, iy - 1, iz] +
                   own_subbox[ix, iy, iz + 1] +
                   own_subbox[ix, iy, iz - 1]
                )
            end
        end
    end

    # syncronise all nodes
    MPI.Barrier(comm)
end

# clean up
MPI.Finalize()


