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

struct Neighbor
    rank::Int
    offset::Tuple{Int, Int, Int}
    receive_buffer::Array{Float64, 3}
end

Base.@kwdef mutable struct MPIState
    state::State
    p::Parameter
    lnx::Int
    lny::Int
    lnz::Int
    neighbors::Vector{Neighbor}
    comm::MPI.Comm
    rank::Int
    requests::Vector{MPI.Request}
end

struct FieldGenerator
    plan :: PencilFFTPlan{Complex{Float64}, 3}
    hat :: PencilArray{Complex{Float64}, 3}
    global_hat :: GlobalPencilArray{Complex{Float64}, 3}
    # plan :: PencilFFTPlan{Float64, 3}
    # hat :: PencilArray{Float64, 3}
    # global_hat :: GlobalPencilArray{Float64, 3}
    ks :: Frequencies{Float64}
end

function FieldGenerator(pen :: Pencil, p :: Parameter)
    # transform = Transforms.RFFT()
    transform = Transforms.FFT()
    plan = PencilFFTPlan(pen, transform)
    hat = allocate_output(plan)
    # ks = AbstractFFTs.rfftfreq(p.N, 1 / p.dx) .* (2*pi)
    ks = AbstractFFTs.fftfreq(p.N, 1 / p.dx) .* (2*pi)
    global_hat = global_view(hat)
    return FieldGenerator(plan, hat, global_hat, ks)
end

function random_field_mpi(field_generator :: FieldGenerator, p :: Parameter)
    @inbounds @simd for i in CartesianIndices(field_generator.global_hat)
        ix, iy, iz = Tuple(i)
        kx = field_generator.ks[ix]
        ky = field_generator.ks[iy]
        kz = field_generator.ks[iz]
        k = sqrt(kx^2 + ky^2 + kz^2)
        field_generator.global_hat[ix, iy, iz] =
            # k <= p.k_max ? (rand()*2 - 1) * field_max : 0.0
            complex(k <= p.k_max ? (rand()*2 - 1) * field_max : 0.0)
    end

    field = allocate_input(field_generator.plan)

    ldiv!(field, field_generator.plan, field_generator.hat)

    # normalize to the local mean
    field ./= mean(abs, parent(field))

    return field
end

function init_state_mpi(p::Parameter)
    # mpi setup
    MPI.Init()
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    print("begin setup $rank\n")

    # setup the grid (divide the simulation into subboxes for each node)
    global_grid_dimension = (p.N, p.N, p.N)

    # use PencilArrays to construct the grid
    pen = Pencil(global_grid_dimension, comm)

    # list of the number of subboxes per dimension
    axis_lengths = tuple(1, size(topology(pen))...)

    # list of index ranges for each node
    remote_shapes = [range_remote(pen, i) for i in 1:nprocs]
    # collect all possible index ranges per dimension
    ranges_per_dimension = [sort(collect(Set(getindex.(remote_shapes, i))))
     for i in 1:length(remote_shapes[1])]
    # find the index of the index range (the index of the subbox) in each
    # dimension for each node
    remote_indicies = [[findfirst(rr -> r[dim] == rr, ranges_per_dimension[dim])
                        for dim in 1:length(remote_shapes[1])]
         for r in remote_shapes]

    # map from the id of a node to its cartesian index in the subbox grid
    node_id_to_box_index = Dict(zip(0:nprocs-1, remote_indicies))
    # map from the cartesian index in the subbox grid to the id of the node
    box_index_to_node_id = Dict(zip(remote_indicies, 0:nprocs-1))

    # without PencilArrays:
    # grid = distributed_grid(global_grid_dimension, nprocs)

    # subbox setup
    Random.seed!(p.seed * rank)
    lnx, lny, lnz = size_local(pen)

    # setup the local part of the simulation box
    field_generator = FieldGenerator(pen, p)

    # generate random psi and psi_dot
    pen_psi = random_field_mpi(field_generator, p)
    pen_psi_dot = random_field_mpi(field_generator, p)

    field_generator = nothing # release field generator memory to gc

    # assign them to arrays with a boarder
    psi = Array{Complex{Float64}}(undef, lnx + 2, lny + 2, lnz + 2)
    psi[2:end-1, 2:end-1, 2:end-1] = parent(pen_psi)
    pen_psi = nothing

    psi_dot = Array{Complex{Float64}}(undef, lnx + 2, lny + 2, lnz + 2)
    psi_dot[2:end-1, 2:end-1, 2:end-1] = parent(pen_psi_dot)
    pen_psi_dot = nothing

    # construct local state object
    own_s = State(p.tau_start, 0, psi, psi_dot,
                  Array{Complex{Float64}}(undef, lnx, lny, lnz),
                  Array{Complex{Float64}}(undef, lnx, lny, lnz),)

    # setup for exchanging data with neighboring subboxes/nodes
    own_subbox_index = node_id_to_box_index[rank]

    # find all neighbors of our subbox and the ranks they belong to
    neighbors = Neighbor[]
    for dim in 1:3
        for side in (1, -1)
            offset = [0, 0, 0]
            offset[dim] = side

            neighbor_index = mod1.(own_subbox_index .+ offset, axis_lengths)
            neighbor_id = box_index_to_node_id[neighbor_index]

            rcvbuf = Array{Float64}(undef,
                                    length(get_receive_index(offset[1], lnx)),
                                    length(get_receive_index(offset[2], lny)),
                                    length(get_receive_index(offset[3], lnz)),)

            neighbor = Neighbor(neighbor_id, tuple(offset...), rcvbuf)

            push!(neighbors, neighbor)
        end
    end

    requests = MPI.Request[]

    s = MPIState(state = own_s,
                 p = p,
                 lnx = lnx,
                 lny = lny,
                 lnz = lnz,
                 neighbors = neighbors,
                 comm = comm,
                 rank = rank,
                 requests = requests,
                )

    print("end setup on rank $rank\n")
    MPI.Barrier(comm)
    return s
end

function step_mpi!(s::MPIState)
    if s.rank == 0
        print("$(s.state.step) of $(s.p.nsteps)\n")
    end
    print("exchange data on node $(s.rank)\n")
    # exchange data with neighboring nodes/subboxes
    for neighbor in s.neighbors
        offx, offy, offz = neighbor.offset
        # slice of own data to send to neighbor
        # TODO: do this for all required arrays
        to_send = @view own_subbox[
            get_send_index(offx, s.lnx),
            get_send_index(offy, s.lny),
            get_send_index(offz, s.lnz),
        ]
        send = MPI.Isend(to_send, neighbor.rank, s.state.step, s.comm)
        push!(s.requests, send)

        # receive data from neighbor
        receive = MPI.Irecv!(neighbor.receive_buffer, neighbor.rank, s.state.step, s.comm)
        push!(s.requests, receive)
    end
    MPI.Waitall(s.requests)

    # assign received data to own subbox
    for neighbor in s.neighbors
        offx, offy, offz = neighbor.offset
        ix = get_receive_index(offx, s.lnx)
        iy = get_receive_index(offy, s.lny)
        iz = get_receive_index(offz, s.lnz)
        # TODO: do this for all required arrays
        own_subbox[ix, iy, iz] = neighbor.receive_buffer
    end

    # reuse the rquests list for next time
    empty!(s.requests)

    print("local update on node $rank\n")

    # do local update
    # TODO: real update

    # syncronise all nodes
    MPI.Barrier(s.comm)
end

function finish_mpi!(s::MPIState)
    # clean up
    MPI.Finalize()
end
