import numpy as np
from dataclasses import dataclass
from scipy.fft import fftfreq
import itertools

############################################# parameters #######################################
def log_to_H(l): return 1.0 / np.exp(l)
def H_to_t(H): return 1 / (2*H)
def t_to_H(t): return 1 / (2*t)
def H_to_log(H): return np.log(1/H)
def t_to_tau(t): return 2* np.sqrt(t)
def log_to_tau(log): return t_to_tau(H_to_t(log_to_H(log)))
def t_to_a(t): return np.sqrt(t)
def tau_to_t(tau): return 0.5*(tau)**2
def tau_to_a(tau): return 0.5*tau
def tau_to_log(tau): return H_to_log(t_to_H(tau_to_t(tau)))

def sim_params_from_physical_scale(log_end):
    L = 1 / log_to_H(log_end)
    N = int(np.ceil(L))
    return L, N

@dataclass
class Parameter:
    # simulation domain in time in log units
    log_start : float
    log_end : float
    # comoving length of the simulation box in units of 1/m_r
    L : float
    # number of grid points in one dimension
    N : int
    # time step size in tau (conformal time) in units of 1/m_r
    Delta_tau : float
    # seed for random number generator
    seed : int
    k_max : float
    dx : float
    tau_start : float
    tau_end : float
    tau_span : float
    nsteps : int
    # params for spectrum computation
    nbins : int
    radius : int

def make_parameter(log_start, log_end, Delta_tau, seed, k_max, nbins, radius):
    L, N = sim_params_from_physical_scale(log_end)

    tau_start = log_to_tau(log_start)
    tau_end = log_to_tau(log_end)
    tau_span = tau_end - tau_start

    return Parameter(
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
        nsteps=int(np.ceil(tau_span / Delta_tau)),
        nbins=nbins,
        radius=radius,
   )

######################################## initializing simulation ##########################################
# get the index used to get data from to send to node with
# neighboring subbox
def get_send_index(off, ln):
    if off == -1:
        return [1]
    elif off == 0:
        return range(1, ln)
    elif off == 1:
        return [ln - 1]
    else:
        raise ValueError("invalid offset")

# get the index used to assign data received from node with
# neighboring subbox
def get_receive_index(off, ln):
    if off == -1:
        return [0]
    elif off == 0:
        return range(1, ln)
    elif off == 1:
        return [ln]
    else:
        raise ValueError("invalid offset")

@dataclass
class FieldGenerator:
    plan: PencilFFTPlan
    hat: PencilArray(complex, 3)
    global_hat: GlobalPencilArray(complex, 3)
    ks: np.ndarray

def make_field_generator(pen: Pencil, p: Parameter) -> FieldGenerator:
    transform = Transforms.FFT()
    plan = PencilFFTPlan(pen, transform)
    hat = allocate_output(plan)
    ks = fftfreq(p.N, p.dx) * (2 * np.pi)
    global_hat = global_view(hat)
    return FieldGenerator(plan, hat, global_hat, ks)

field_max = 1 / np.sqrt(2)

def random_field_mpi(field_generator: FieldGenerator, p: Parameter) -> np.ndarray:
    for ix, iy, iz in itertools.product([range(field_generator.global_hat.shape[0]),
                                         range(field_generator.global_hat.shape[1]),
                                         range(field_generator.global_hat.shape[2]),]):
        kx = field_generator.ks[ix]
        ky = field_generator.ks[iy]
        kz = field_generator.ks[iz]
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        field_generator.global_hat[ix, iy, iz] = field_max * np.random.rand() * np.exp(2*np.pi*1j * np.random.rand()) if k <= p.k_max else 0.0
    field = allocate_input(field_generator.plan)
    ldiv(field, field_generator.plan, field_generator.hat)

    # field .*= size(field, 1)**3

    return field

# infos about neighboring subbox
@dataclass
class Neighbor:
    rank:int
    offset: (int,int,int)
    receive_buffer: np.ndarray

@dataclass
class State:
    tau: float
    step: int
    psi: np.ndarray[complex]
    psi_dot: np.ndarray[complex]
    psi_dot_dot: np.ndarray[complex]
    next_psi_dot_dot: np.ndarray[complex]
    lnx: int
    lny: int
    lnz: int
    pen: Pencil
    neighbors: list[Neighbor]
    comm: MPI.Comm
    rank: int
    root: int

######################################### computing the rhs #######################################
def exchange_field(s:State):
    requests: list[MPI.Request] = []
    for neighbor in s.neighbors:
        if neighbor.rank == s.rank:
            # no need for communication -> copy memory directly
            pass
        else:
            # receive data from neighbor
            receive = MPI.Irecv(neighbor.receive_buffer,
                                 s.comm,
                                 source=neighbor.rank,
                                 tag=neighbor.rank)
            requests.append(receive)

            # slice of own data to send to neighbor
            offx, offy, offz = neighbor.offset
            to_send = s.psi[
                get_send_index(offx, s.lnx),
                get_send_index(offy, s.lny),
                get_send_index(offz, s.lnz),
            ]
            send = MPI.Isend(to_send, s.comm, dest=neighbor.rank, tag=s.rank)
            requests.append(send)
    MPI.Waitall(requests)

    # assign received data to own subbox
    for neighbor in s.neighbors:
        offx, offy, offz = neighbor.offset
        ix = get_receive_index(offx, s.lnx)
        iy = get_receive_index(offy, s.lny)
        iz = get_receive_index(offz, s.lnz)
        if neighbor.rank == s.rank:
            s.psi[ix, iy, iz] = s.psi[
                get_send_index(offx, s.lnx),
                get_send_index(offy, s.lny),
                get_send_index(offz, s.lnz),
            ]
        else:
            s.psi[ix, iy, iz] = neighbor.receive_buffer

    # reuse the rquests list for next time

def compute_force(out: np.ndarray[complex], s:State, p:Parameter):
    # exchange data with neighboring nodes/subboxes
    # we need to exchange the psi field as we need to compute its
    # laplacian for time propagation
    exchange_field(s)

    a = tau_to_a(s.tau)
    for ix in range(1, s.lnx):
       for iy in range(1, s.lny):
            for iz in range(1, s.lnz):
                psi = s.psi[ix, iy, iz]
                left = s.psi[ix + 1, iy, iz]
                right = s.psi[ix - 1, iy, iz]
                front = s.psi[ix, iy + 1, iz]
                back = s.psi[ix, iy - 1, iz]
                top = s.psi[ix, iy, iz + 1]
                bottom = s.psi[ix, iy, iz - 1]
                pot_force = psi * (psi*psi.conj() - 0.5*a)
                laplace = (- 6 * psi + left + right + front + back + top + bottom) / p.dx**2
                out[ix, iy, iz] = + laplace - pot_force

    # syncronise all nodes
    MPI.Barrier(s.comm)

###################################### initializing the simulation (cont.) ###############################
def State(p:Parameter):
    # mpi setup
    MPI.Init()
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

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
    axis_lengths = (1,) + np.shape(topology(pen))
    if rank == 0:
        print(pen)
        print(topology(pen))

    # list of index ranges for each node
    remote_shapes = [range_remote(pen, i) for i in range(nprocs)]
    # collect all possible index ranges per dimension
    ranges_per_dimension = [sorted(set([remote_shapes[j] for j in range(len(remote_shapes))])) for i in range(len(remote_shapes))]
    # find the index of the index range (the index of the subbox) in each
    # dimension for each node
    remote_indicies = [[filter(lambda rr: r[dim] == rr, ranges_per_dimension[dim])[0]
                        for dim in range(len(remote_shapes[0]))] for r in remote_shapes]

    # map from the id of a node to its cartesian index in the subbox grid
    node_id_to_box_index = dict(enumerate(remote_indicies))
    # map from the cartesian index in the subbox grid to the id of the node
    box_index_to_node_id = dict(zip(remote_indicies, range(nprocs)))

    # without PencilArrays:
    # grid = distributed_grid(global_grid_dimension, nprocs)

    # subbox setup
    np.random.seed(p.seed + rank)
    lnx, lny, lnz = size_local(pen)

    # setup the local part of the simulation box
    field_generator = make_field_generator(pen, p)

    # generate random psi and psi_dot
    # we are doing this is akward way to save on memory
    pen_psi = random_field_mpi(field_generator, p) # initialy phi
    pen_psi_dot = random_field_mpi(field_generator, p) # initialy d phi / dt

    if rank == root:
        print(sum(np.abs(pen_psi)**2) / np.shape(pen_psi, 0)**3)

    H = t_to_H(tau_to_t(p.tau_start))
    a = tau_to_a(p.tau_start)

    pen_psi_dot += H * pen_psi # here pen_psi is still phi
    pen_psi_dot *= a**2
    pen_psi *= a

    field_generator = None # release field generator memory to gc

    # assign them to arrays with a boarder
    psi = np.empty((lnx + 2, lny + 2, lnz + 2), dtype=np.complex64)
    psi[1:-1, 1:-1, 1:-1] = parent(pen_psi)
    pen_psi = None

    psi_dot = np.empty((lnx + 2, lny + 2, lnz + 2), dtype=np.complex64)
    psi_dot[1:-1, 1:-1, 1:-1] = parent(pen_psi_dot)
    pen_psi_dot = None

    # setup for exchanging data with neighboring subboxes/nodes
    own_subbox_index = node_id_to_box_index[rank]

    # find all neighbors of our subbox and the ranks they belong to
    neighbors = []
    for dim in range(3):
        for side in (1, -1):
            offset = np.array([0, 0, 0])
            offset[dim] = side

            neighbor_index = (own_subbox_index + offset) % axis_lengths
            neighbor_id = box_index_to_node_id[neighbor_index]

            rcvbuf = np.empty((
                len(get_receive_index(offset[0], lnx)),
                len(get_receive_index(offset[1], lny)),
                len(get_receive_index(offset[2], lnz))))

            neighbor = Neighbor(neighbor_id, tuple(offset), rcvbuf)

            neighbors.append(neighbor)

    s = State(
        tau=p.tau_start,
        step=0,
        psi=psi,
        psi_dot=psi_dot,
        psi_dot_dot=np.empty((lnx, lny, lnz)),
        next_psi_dot_dot=np.empty((lnx, lny, lnz)),
        lnx=lnx,
        lny=lny,
        lnz=lnz,
        pen=pen,
        neighbors=neighbors,
        comm=comm,
        rank=rank,
        root=root,
    )

    MPI.Barrier(comm)

    compute_force(s.psi_dot_dot, s, p)

    print(f"end setup on rank {rank}")

    return s

def finish_mpi():
    MPI.Finalize()

############################################# time stepping ######################################
def step(s:State, p:Parameter):
    if s.rank == 0:
        print(f"{s.step} of {p.nsteps}")
    print(f"exchange data on node {s.rank}")

    # do local update
    print(f"local update on node {s.rank}\n")

    # this is the method used by gorghetto in axion strings: the attractive solution
    # propagate PDE using velocity verlet algorithm
    s.tau = p.tau_start + (s.step + 1) * p.Delta_tau
    s.step += 1

    # get range of the arrays to be updated
    update_psi = s.psi[1:-1, 1:-1, 1:-1]
    update_psi_dot = s.psi_dot[1:-1, 1:-1, 1:-1]

    # update the field ("position")
    update_psi += p.Delta_tau*update_psi_dot + 0.5*p.Delta_tau**2*s.psi_dot_dot

    # update the field derivative ("velocity")
    compute_force(s.next_psi_dot_dot, s, p)
    update_psi_dot += p.Delta_tau*0.5*(s.psi_dot_dot + s.next_psi_dot_dot)

    # swap current and next arrays
    (s.psi_dot_dot, s.next_psi_dot_dot) = (s.next_psi_dot_dot, s.psi_dot_dot)
    print(f"end frame {s.step} on rank {s.rank}")

    # syncronise all nodes
    MPI.Barrier(s.comm)
