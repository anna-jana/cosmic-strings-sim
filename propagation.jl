function compute_force!(out :: Array{Complex{Float64}, 3}, s :: State, p :: Parameter)
    scale_factor = tau_to_a(s.tau)
    Threads.@threads for iz in 1:p.N
        for iy in 1:p.N
            @simd for ix in 1:p.N
                @inbounds pot_force = s.psi[ix, iy, iz] * (abs2(s.psi[ix, iy, iz]) - 0.5*scale_factor)
                @inbounds laplace = (- 6 * s.psi[ix, iy, iz] +
                    s.psi[mod1(ix + 1, p.N), iy, iz] +
                    s.psi[mod1(ix - 1, p.N), iy, iz] +
                    s.psi[ix, mod1(iy + 1, p.N), iz] +
                    s.psi[ix, mod1(iy - 1, p.N), iz] +
                    s.psi[ix, iy, mod1(iz + 1, p.N)] +
                    s.psi[ix, iy, mod1(iz - 1, p.N)]) / p.dx^2
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
    Threads.@threads for i in eachindex(s.psi)
        @inbounds s.psi[i] += p.Delta_tau*s.psi_dot[i] + 0.5*p.Delta_tau^2*s.psi_dot_dot[i]
    end

    # update the field derivative ("velocity")
    compute_force!(s.next_psi_dot_dot, s, p)

    Threads.@threads for i in eachindex(s.psi_dot)
        @inbounds s.psi_dot[i] += p.Delta_tau*0.5*(s.psi_dot_dot[i] + s.next_psi_dot_dot[i])
    end

    # swap current and next arrays
    (s.psi_dot_dot, s.next_psi_dot_dot) = (s.next_psi_dot_dot, s.psi_dot_dot)
end
