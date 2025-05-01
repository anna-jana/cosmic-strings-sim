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

######################################## simulation utils ##########################################
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
        return self.plan.backward(self.hat) / self.p.N

# infos about neighboring subbox
@dataclass
class Neighbor:
    rank: int
    offset: (int,int,int)
    receive_buffer: np.ndarray

######################################### computing the rhs #######################################
@numba.njit
def compute_force_local(out, psi_array, lnx, lny, lnz, dx, a):
    for ix in range(1, lnx + 1):
       for iy in range(1, lny + 1):
            for iz in range(1, lnz + 1):
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
                if not np.isfinite(out[ix, iy, iz]):
                    raise ValueError("instable simulation")

########################################## simulation state #######################################
class State:
    def __init__(self, p: Parameter):
        # mpi setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        root = 0
        self.p = p

        print(f"begin setup of rank {rank}")

        # setup the grid (divide the simulation into subboxes for each node)
        global_grid_dimension = (self.p.N, self.p.N, self.p.N)

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

        # subbox setup
        np.random.seed(self.p.seed + rank)
        lnx, lny, lnz = darray.shape

        # setup the local part of the simulation box
        field_generator = FieldGenerator(self.p)

        # generate random psi and psi_dot
        # we are doing this is akward way to save on memory
        pen_psi = field_generator.random_field_mpi() # initialy phi
        pen_psi_dot = field_generator.random_field_mpi() # initialy d phi / dt

        H = t_to_H(tau_to_t(self.p.tau_start))
        a = tau_to_a(self.p.tau_start)

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

        self.tau = self.p.tau_start
        self.step = 0
        self.psi = psi
        self.psi_dot = psi_dot
        self.psi_dot_dot = np.empty((lnx + 2, lny + 2, lnz + 2), dtype = np.complex128)
        self.next_psi_dot_dot = np.empty((lnx + 2, lny + 2, lnz + 2), dtype = np.complex128)
        self.lnx = lnx
        self.lny = lny
        self.lnz = lnz
        self.pen = darray
        self.neighbors = neighbors
        self.comm = comm
        self.rank = rank
        self.root = root

        comm.Barrier()

        self.compute_force(self.psi_dot_dot)

        print(f"end setup of rank {rank}")
        if rank == root:
            print("being simuölation loop")

    def finish_mpi(self):
        MPI.Finalize()

    def exchange_field(self):
        requests: list[MPI.Request] = []
        for neighbor in self.neighbors:
            if neighbor.rank == self.rank:
                # no need for communication -> copy memory directly
                pass
            else:
                # receive data from neighbor
                receive = self.comm.Irecv([neighbor.receive_buffer, MPI.COMPLEX16], source=neighbor.rank)
                requests.append(receive)

                # slice of own data to send to neighbor
                offx, offy, offz = neighbor.offset
                send_start_x, send_stop_x = get_send_index(offx, self.lnx)
                send_start_y, send_stop_y = get_send_index(offy, self.lny)
                send_start_z, send_stop_z = get_send_index(offz, self.lnz)
                to_send = self.psi[send_start_x:send_stop_x, send_start_y:send_stop_y, send_start_z:send_stop_z].copy()
                send = self.comm.Isend([to_send, MPI.COMPLEX16], dest=neighbor.rank)
                requests.append(send)

            MPI.Request.Waitall(requests)

        # assign received data to own subbox
        for neighbor in self.neighbors:
            offx, offy, offz = neighbor.offset
            receive_start_x, receive_stop_x = get_receive_index(offx, self.lnx)
            receive_start_y, receive_stop_y = get_receive_index(offy, self.lny)
            receive_start_z, receive_stop_z = get_receive_index(offz, self.lnz)
            if neighbor.rank == self.rank:
                send_start_x, send_stop_x = get_send_index(offx, self.lnx)
                send_start_y, send_stop_y = get_send_index(offy, self.lny)
                send_start_z, send_stop_z = get_send_index(offz, self.lnz)
                self.psi[receive_start_x:receive_stop_x, receive_start_y:receive_stop_y, receive_start_z:receive_stop_z] = self.psi[
                    send_start_x:send_stop_x, send_start_y:send_stop_y, send_start_z:send_stop_z
                ]
            else:
                self.psi[receive_start_x:receive_stop_x, receive_start_y:receive_stop_y, receive_start_z:receive_stop_z] = neighbor.receive_buffer


        # reuse the rquests list for next time

    def compute_force(self, out: np.ndarray[complex]):
        # exchange data with neighboring nodes/subboxes
        # we need to exchange the psi field as we need to compute its
        # laplacian for time propagation
        self.exchange_field()

        # compute the force for the local array
        compute_force_local(out, self.psi, self.lnx, self.lny, self.lnz, self.p.dx, tau_to_a(self.tau))

        # syncronise all nodes
        self.comm.Barrier()

    ############################################# time stepping ######################################
    def do_step(self):
        if self.rank == 0:
            print(f"{self.step} of {self.p.nsteps}")

        # this is the method used by gorghetto in axion strings: the attractive solution
        # propagate PDE using velocity verlet algorithm
        self.step += 1

        # get range of the arrays to be updated
        update_psi = self.psi[1:-1, 1:-1, 1:-1]
        update_psi_dot = self.psi_dot[1:-1, 1:-1, 1:-1]

        # update the field ("position")
        update_psi += self.p.Delta_tau*update_psi_dot + 0.5 * self.p.Delta_tau**2 * self.psi_dot_dot[1:-1, 1:-1, 1:-1]

        # update the field derivative ("velocity")
        self.tau += self.p.Delta_tau / 2.0
        self.compute_force(self.next_psi_dot_dot)
        self.tau += self.p.Delta_tau / 2.0
        update_psi_dot += self.p.Delta_tau * 0.5 * (self.psi_dot_dot[1:-1, 1:-1, 1:-1] + self.next_psi_dot_dot[1:-1, 1:-1, 1:-1])

        # swap current and next arrays
        (self.psi_dot_dot, self.next_psi_dot_dot) = (self.next_psi_dot_dot, self.psi_dot_dot)

        diagnostic = self.comm.reduce(np.sum(np.abs(update_psi)), op=MPI.SUM, root=self.root)
        if self.rank == self.root:
            print("mean magnitude:", diagnostic / self.p.N**3)

        # syncronise all nodes
        self.comm.Barrier()

