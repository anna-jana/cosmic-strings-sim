function compute_force!(out :: Array{Complex{Float64}, 3}, s :: State, p :: Parameter)
    scale_factor = tau_to_a(s.tau)
    Threads.@threads for iz in 1:p.N
        for iy in 1:p.N
            @simd for ix in 1:p.N
                @inbounds pot_force = s.phi[ix, iy, iz] * (abs2(s.phi[ix, iy, iz]) - 0.5*scale_factor)
                @inbounds laplace = (- 6 * s.phi[ix, iy, iz] +
                    s.phi[mod1(ix + 1, p.N), iy, iz] +
                    s.phi[mod1(ix - 1, p.N), iy, iz] +
                    s.phi[ix, mod1(iy + 1, p.N), iz] +
                    s.phi[ix, mod1(iy - 1, p.N), iz] +
                    s.phi[ix, iy, mod1(iz + 1, p.N)] +
                    s.phi[ix, iy, mod1(iz - 1, p.N)]) / p.dx^2
                @inbounds out[ix, iy, iz] = + laplace - pot_force
            end
        end
    end
end

function make_step!(s :: State, p :: Parameter)
    # propagate PDE using velocity verlet algorithm
    s.tau = p.tau_start + (s.step + 1) * p.Delta_tau
    s.step += 1

    # update the field ("position")
    Threads.@threads for i in eachindex(s.phi)
        @inbounds s.phi[i] += p.Delta_tau*s.phi_dot[i] + 0.5*p.Delta_tau^2*s.phi_dot_dot[i]
    end

    # update the field derivative ("velocity")
    compute_force!(s.next_phi_dot_dot, s, p)

    Threads.@threads for i in eachindex(s.phi_dot)
        @inbounds s.phi_dot[i] += p.Delta_tau*0.5*(s.phi_dot_dot[i] + s.next_phi_dot_dot[i])
    end

    # swap current and next arrays
    (s.phi_dot_dot, s.next_phi_dot_dot) = (s.next_phi_dot_dot, s.phi_dot_dot)
end

function run_simulation!(s::State, p::Parameter)
    for i in 1:p.nsteps
        println("$i of $(p.nsteps)")
        make_step!(s, p)
    end
end

function run_simulation!(callback::Function, s::State, p::Parameter, ntimes::Int64)
    every = div(p.nsteps, ntimes)
    for i in 1:p.nsteps
        println("$i of $(p.nsteps)")
        if i % every == 0
            callback()
        end
        make_step!(s, p)
    end
    if p.nsteps % every != 0
        callback()
    end
end

