@inline function force_stecil(psi, left, right, front, back, top, bottom,
                              a, p::Parameter)
    pot_force = psi * (abs2(psi) - 0.5*a)
    laplace = (- 6 * psi + left + right + front + back + top + bottom) / p.dx^2
    return + laplace - pot_force
end

function make_step_single_box!(s :: State, p :: Parameter)
    # propagate PDE using velocity verlet algorithm
    s.tau = p.tau_start + (s.step + 1) * p.Delta_tau
    s.step += 1

    # update the field ("position")
    Threads.@threads for i in eachindex(s.psi)
        @inbounds s.psi[i] += p.Delta_tau*s.psi_dot[i] +
                              0.5*p.Delta_tau^2*s.psi_dot_dot[i]
    end

    # update the field derivative ("velocity")
    compute_force_single_node!(s.next_psi_dot_dot, s, p)

    Threads.@threads for i in eachindex(s.psi_dot)
        @inbounds s.psi_dot[i] += p.Delta_tau*0.5*(s.psi_dot_dot[i] + s.next_psi_dot_dot[i])
    end

    # swap current and next arrays
    (s.psi_dot_dot, s.next_psi_dot_dot) = (s.next_psi_dot_dot, s.psi_dot_dot)
end
