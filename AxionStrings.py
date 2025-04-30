from dataclasses import dataclass
import itertools
import numpy as np
from scipy.fft import fftfreq
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray, DistArray
import numba

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
        return 1, 2
    elif off == 0:
        return 1, ln
    elif off == 1:
        return ln - 1, ln
    else:
        raise ValueError("invalid offset")

# get the index used to assign data received from node with
# neighboring subbox
def get_receive_index(off, ln):
    if off == -1:
        return 0, 1
    elif off == 0:
        return 1, ln
    elif off == 1:
        return ln, ln + 1
    else:
        raise ValueError("invalid offset")

def iter_global_indicies(d: DistArray):
    return itertools.product(*[list(range(start, start + size)) for start, size in zip(d.substart, d.shape)])

@dataclass
class FieldGenerator:
    def __init__(self, p: Parameter):
        self.p = p
        self.ks = fftfreq(p.N, p.dx) * (2 * np.pi)
        self.plan = PFFT(MPI.COMM_WORLD, (p.N, p.N, p.N), axes=(0, 1, 2), dtype=np.complex128, backend="scipy")
        self.hat = newDistArray(self.plan, True)
        self.field_max = 1 / np.sqrt(2)

    def random_field_mpi(self) -> np.ndarray:
        sx, sy, sz = self.hat.substart
        for ix, iy, iz in iter_global_indicies(self.hat):
            kx = self.ks[ix]
            ky = self.ks[iy]
            kz = self.ks[iz]
            k = np.sqrt(kx**2 + ky**2 + kz**2)
            self.hat[ix - sx, iy - sy, iz - sz] = (
                    self.field_max * np.random.rand() * np.exp(2*np.pi*1j * np.random.rand())
                    if k <= self.p.k_max else 0.0
            )
        return self.plan.backward(self.hat)

# infos about neighboring subbox
@dataclass
class Neighbor:
    rank: int
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
    pen: DistArray
    neighbors: list[Neighbor]
    comm: MPI.Comm
    rank: int
    root: int

######################################### computing the rhs #######################################
def compute_message_tag(s, sender_rank, receiver_rank):
    nranks = s.comm.Get_size()
    return sender_rank + receiver_rank*nranks

def exchange_field(s:State):
    requests: list[MPI.Request] = []
    for neighbor in s.neighbors:
        if neighbor.rank == s.rank:
            # no need for communication -> copy memory directly
            pass
        else:
            # receive data from neighbor
            receive = s.comm.Irecv([neighbor.receive_buffer, MPI.COMPLEX16], source=neighbor.rank) # , tag=compute_message_tag(s, neighbor.rank, s.rank))
            requests.append(receive)

            # slice of own data to send to neighbor
            offx, offy, offz = neighbor.offset
            send_start_x, send_stop_x = get_send_index(offx, s.lnx)
            send_start_y, send_stop_y = get_send_index(offy, s.lny)
            send_start_z, send_stop_z = get_send_index(offz, s.lnz)
            to_send = s.psi[send_start_x:send_stop_x, send_start_y:send_stop_y, send_start_z:send_stop_z].copy()
            send = s.comm.Isend([to_send, MPI.COMPLEX16], dest=neighbor.rank) # , tag=compute_message_tag(s, s.rank, neighbor.rank))
            requests.append(send)

        MPI.Request.Waitall(requests)

    # assign received data to own subbox
    for neighbor in s.neighbors:
        offx, offy, offz = neighbor.offset
        receive_start_x, receive_stop_x = get_receive_index(offx, s.lnx)
        receive_start_y, receive_stop_y = get_receive_index(offy, s.lny)
        receive_start_z, receive_stop_z = get_receive_index(offz, s.lnz)
        if neighbor.rank == s.rank:
            send_start_x, send_stop_x = get_send_index(offx, s.lnx)
            send_start_y, send_stop_y = get_send_index(offy, s.lny)
            send_start_z, send_stop_z = get_send_index(offz, s.lnz)
            s.psi[receive_start_x:receive_stop_x, receive_start_y:receive_stop_y, receive_start_z:receive_stop_z] = s.psi[
                send_start_x:send_stop_x, send_start_y:send_stop_y, send_start_z:send_stop_z
            ]
        else:
            s.psi[receive_start_x:receive_stop_x, receive_start_y:receive_stop_y, receive_start_z:receive_stop_z] = neighbor.receive_buffer


    # reuse the rquests list for next time

@numba.njit
def compute_force_local(out, psi_array, lnx, lny, lnz, dx, a):
    for ix in range(1, lnx):
       for iy in range(1, lny):
            for iz in range(1, lnz):
                left   = psi_array[ix + 1, iy, iz]
                right  = psi_array[ix - 1, iy, iz]
                front  = psi_array[ix, iy + 1, iz]
                back   = psi_array[ix, iy - 1, iz]
                top    = psi_array[ix, iy, iz + 1]
                bottom = psi_array[ix, iy, iz - 1]
                psi = psi_array[ix, iy, iz]
                pot_force = psi * (psi*np.conj(psi) - 0.5*a)
                laplace = (- 6 * psi + left + right + front + back + top + bottom) / dx**2
                out[ix, iy, iz] = + laplace - pot_force

def compute_force(out: np.ndarray[complex], s:State, p:Parameter):
    # exchange data with neighboring nodes/subboxes
    # we need to exchange the psi field as we need to compute its
    # laplacian for time propagation
    exchange_field(s)

    # compute the force for the local array
    compute_force_local(out, s.psi, s.lnx, s.lny, s.lnz, p.dx, tau_to_a(s.tau))

    # syncronise all nodes
    s.comm.Barrier()

###################################### initializing the simulation (cont.) ###############################
def make_state(p: Parameter):
    # mpi setup
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    root = 0

    print(f"begin setup {rank}\n")

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
    darray = DistArray(global_grid_dimension)

    # list of the number of subboxes per dimension
    axis_lengths = darray.commsizes

    # list of index ranges for each node
    remote_ranks = comm.allgather(comm.Get_rank())
    remote_slices = comm.allgather(darray.local_slice())

    axis_starts = [list(sorted(set(remote_slice[i].start for remote_slice in remote_slices))) for i in range(3)]
    remote_box_indices = [tuple(axis_starts[i].index(remote_slice[i].start) for i in range(3)) for remote_slice in remote_slices]

    # map from the id of a node to its cartesian index in the subbox grid
    node_id_to_box_index = dict(zip(remote_ranks, remote_box_indices))
    # map from the cartesian index in the subbox grid to the id of the node
    box_index_to_node_id = dict(zip(remote_box_indices, remote_ranks))

    if rank == root:
        print("ranks:", remote_ranks)
        print("slices on each rank:", remote_slices)
        print("start indices for the boxes on each axis:", axis_starts)
        print("box index for each rank:", remote_box_indices)

    # without PencilArrays:
    # grid = distributed_grid(global_grid_dimension, nprocs)

    # subbox setup
    np.random.seed(p.seed + rank)
    lnx, lny, lnz = darray.shape

    # setup the local part of the simulation box
    field_generator = FieldGenerator(p)

    # generate random psi and psi_dot
    # we are doing this is akward way to save on memory
    pen_psi = field_generator.random_field_mpi() # initialy phi
    pen_psi_dot = field_generator.random_field_mpi() # initialy d phi / dt

    H = t_to_H(tau_to_t(p.tau_start))
    a = tau_to_a(p.tau_start)

    pen_psi_dot += H * pen_psi # here pen_psi is still phi
    pen_psi_dot *= a**2
    pen_psi *= a

    field_generator = None # release field generator memory to gc

    # assign them to arrays with a boarder
    psi = np.empty((lnx + 2, lny + 2, lnz + 2), dtype=np.complex128)
    psi[1:-1, 1:-1, 1:-1] = pen_psi
    pen_psi = None

    psi_dot = np.empty((lnx + 2, lny + 2, lnz + 2), dtype=np.complex128)
    psi_dot[1:-1, 1:-1, 1:-1] = pen_psi_dot
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
            neighbor_id = box_index_to_node_id[tuple(neighbor_index)]

            receive_start_x, receive_stop_x = get_receive_index(offset[0], lnx)
            receive_start_y, receive_stop_y = get_receive_index(offset[1], lny)
            receive_start_z, receive_stop_z = get_receive_index(offset[2], lnz)
            rcvbuf = np.empty((receive_stop_x - receive_start_x, receive_stop_y - receive_start_y, receive_stop_z - receive_start_z), dtype=np.complex128)

            neighbor = Neighbor(neighbor_id, tuple(offset), rcvbuf)

            neighbors.append(neighbor)

    s = State(
        tau=p.tau_start,
        step=0,
        psi=psi,
        psi_dot=psi_dot,
        psi_dot_dot=np.empty((lnx, lny, lnz), dtype=np.complex128),
        next_psi_dot_dot=np.empty((lnx, lny, lnz), dtype=np.complex128),
        lnx=lnx,
        lny=lny,
        lnz=lnz,
        pen=darray,
        neighbors=neighbors,
        comm=comm,
        rank=rank,
        root=root,
    )

    comm.Barrier()

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
    s.comm.Barrier()

