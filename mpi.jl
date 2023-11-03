# get the index used to get data from to send to node with
# neighboring subbox
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

# get the index used to assign data received from node with
# neighboring subbox
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

# infos about neighboring subbox
struct Neighbor
    rank::Int
    offset::Tuple{Int,Int,Int}
    receive_buffer::Array{Complex{Float64}, 3}
end

Base.@kwdef mutable struct MPIState <: AbstractState
    tau::Float64
    step::Int
    psi::Array{Complex{Float64}, 3}
    psi_dot::Array{Complex{Float64}, 3}
    psi_dot_dot::Array{Complex{Float64}, 3}
    next_psi_dot_dot::Array{Complex{Float64}, 3}
    lnx::Int
    lny::Int
    lnz::Int
    pen::Pencil
    neighbors::Vector{Neighbor}
    comm::MPI.Comm
    rank::Int
    requests::Vector{MPI.Request}
    root::Int
end

struct FieldGenerator
    plan::PencilFFTPlan{Complex{Float64}, 3}
    hat::PencilArray{Complex{Float64}, 3}
    global_hat::GlobalPencilArray{Complex{Float64}, 3}
    # plan :: PencilFFTPlan{Float64, 3}
    # hat :: PencilArray{Float64, 3}
    # global_hat :: GlobalPencilArray{Float64, 3}
    ks::Frequencies{Float64}
end

function FieldGenerator(pen::Pencil, p::Parameter)
    # transform = Transforms.RFFT()
    transform = Transforms.FFT()
    plan = PencilFFTPlan(pen, transform)
    hat = allocate_output(plan)
    # ks = AbstractFFTs.rfftfreq(p.N, 1 / p.dx) .* (2*pi)
    ks = AbstractFFTs.fftfreq(p.N, 1 / p.dx) .* (2 * pi)
    global_hat = global_view(hat)
    return FieldGenerator(plan, hat, global_hat, ks)
end

function random_field_mpi(field_generator::FieldGenerator, p::Parameter)
    @inbounds @simd for i in CartesianIndices(field_generator.global_hat)
        ix, iy, iz = Tuple(i)
        kx = field_generator.ks[ix]
        ky = field_generator.ks[iy]
        kz = field_generator.ks[iz]
        k = sqrt(kx^2 + ky^2 + kz^2)
        field_generator.global_hat[ix, iy, iz] =
        # k <= p.k_max ? (rand()*2 - 1) * field_max : 0.0
            complex(k <= p.k_max ? (rand() * 2 - 1) * field_max : 0.0)
    end

    field = allocate_input(field_generator.plan)

    ldiv!(field, field_generator.plan, field_generator.hat)

    # normalize to the local mean
    field ./= mean(abs, parent(field))

    return field
end

function MPIState(p::Parameter)
    # mpi setup
    MPI.Init()
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    print("begin setup $rank\n")

    # setup the grid (divide the simulation into subboxes for each node)
    global_grid_dimension = (p.N, p.N, p.N)

    # use PencilArrays to construct the grid
    # Normally splitting the grid into subboxes for each node
    # such that the required communication is minimal is best.
    # That means than i.e. the sum of surfaces of each subbox is minimal
    # i.e. each subbox is as close to a cube as possible.
    # However we also need to take ffts for initalisation and spectrum
    # computation. The Pencil FFT lib only supports pencil configurations
    # i.e. decompositions where the first dimension is not divided, as this
    # is the optimal decomposition for ffts.
    # We need to see if this works for us. If not, we need to find a way to take
    # ffts on a cube decomposition. Either this means that we need to
    # find a different mpi fft library which supports this kind of decomposition
    # or we need to redistribute the data before/after ffts.
    # The later is propbably expensive and complicated.
    pen = Pencil(global_grid_dimension, comm)

    # list of the number of subboxes per dimension
    axis_lengths = tuple(1, size(topology(pen))...)
    if rank == 0
        println(pen)
        println(topology(pen))
    end

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

    root = 0

    s = MPIState(
        tau=p.tau_start,
        step=0,
        psi=psi,
        psi_dot=psi_dot,
        psi_dot_dot=Array{Complex{Float64}}(undef, lnx, lny, lnz),
        next_psi_dot_dot=Array{Complex{Float64}}(undef, lnx, lny, lnz),
        lnx=lnx,
        lny=lny,
        lnz=lnz,
        pen=pen,
        neighbors=neighbors,
        comm=comm,
        rank=rank,
        requests=requests,
        root=root,
    )

    MPI.Barrier(comm)

    compute_force!(s.psi_dot_dot, s, p)

    print("end setup on rank $rank\n")

    return s
end

@inline function get_update_domain(_s::MPIState, A::Array{Complex{Float64}, 3})
    _s # ignore
    return @view A[2:end-1, 2:end-1, 2:end-1]
end

function exchange_field!(s::MPIState)
    for neighbor in s.neighbors
        if neighbor.rank == s.rank
            # no need for communication -> copy memory directly
        else
            # receive data from neighbor
            receive = MPI.Irecv!(neighbor.receive_buffer,
                                 s.comm;
                                 source=neighbor.rank,
                                 tag=neighbor.rank)
            push!(s.requests, receive)

            # slice of own data to send to neighbor
            offx, offy, offz = neighbor.offset
            to_send = @view s.psi[
                get_send_index(offx, s.lnx),
                get_send_index(offy, s.lny),
                get_send_index(offz, s.lnz),
            ]
            send = MPI.Isend(to_send, s.comm; dest=neighbor.rank, tag=s.rank)
            push!(s.requests, send)
        end
    end
    MPI.Waitall(s.requests)

    # assign received data to own subbox
    for neighbor in s.neighbors
        offx, offy, offz = neighbor.offset
        ix = get_receive_index(offx, s.lnx)
        iy = get_receive_index(offy, s.lny)
        iz = get_receive_index(offz, s.lnz)
        if neighbor.rank == s.rank
            s.psi[ix, iy, iz] = @view s.psi[
                get_send_index(offx, s.lnx),
                get_send_index(offy, s.lny),
                get_send_index(offz, s.lnz),
            ]
        else
            s.psi[ix, iy, iz] = neighbor.receive_buffer
        end
    end

    # reuse the rquests list for next time
    empty!(s.requests)
end

function compute_force!(out::Array{Complex{Float64}, 3}, s::MPIState, p::Parameter)
    # exchange data with neighboring nodes/subboxes
    # we need to exchange the psi field as we need to compute its
    # laplacian for time propagation
    exchange_field!(s)

    a = tau_to_a(s.tau)
    @inbounds for iz in 2:s.lnz
        for iy in 2:s.lny
            @simd for ix in 2:s.lnx
                psi = s.psi[ix, iy, iz]
                left = s.psi[ix + 1, iy, iz]
                right = s.psi[ix - 1, iy, iz]
                front = s.psi[ix, iy + 1, iz]
                back = s.psi[ix, iy - 1, iz]
                top = s.psi[ix, iy, iz + 1]
                bottom = s.psi[ix, iy, iz - 1]
                out[ix, iy, iz] = force_stecil(
                    psi, left, right, front, back, top, bottom, a, p)
            end
        end
    end

    # syncronise all nodes
    MPI.Barrier(s.comm)
end

function step_mpi!(s::MPIState, p::Parameter)
    if s.rank == 0
        print("$(s.step) of $(p.nsteps)\n")
    end
    print("exchange data on node $(s.rank)\n")

    # do local update
    print("local update on node $rank\n")
    make_step!(s, p)
    print("end frame $(s.step) on rank $(s.rank)\n")

    # syncronise all nodes
    MPI.Barrier(s.comm)
end

function finish_mpi!(_s::MPIState)
    _s # ignore
    # clean up
    MPI.Finalize()
end
