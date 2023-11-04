@inline function force_stecil(psi, left, right, front, back, top, bottom,
                              a, p::Parameter)
    pot_force = psi * (abs2(psi) - 0.5*a)
    laplace = (- 6 * psi + left + right + front + back + top + bottom) / p.dx^2
    return + laplace - pot_force
end

function make_step!(s::AbstractState, p::Parameter)
    # this is the method used by gorghetto in axion strings: the attractive solution
    # propagate PDE using velocity verlet algorithm
    s.tau = p.tau_start + (s.step + 1) * p.Delta_tau
    s.step += 1

    # get range of the arrays to be updated
    update_psi = get_update_domain(s, s.psi)
    update_psi_dot = get_update_domain(s, s.psi_dot)

    # update the field ("position")
    Threads.@threads for i in eachindex(update_psi)
        @inbounds update_psi[i] += p.Delta_tau*update_psi_dot[i] + 0.5*p.Delta_tau^2*s.psi_dot_dot[i]
    end

    # update the field derivative ("velocity")
    compute_force!(s.next_psi_dot_dot, s, p)

    Threads.@threads for i in eachindex(update_psi_dot)
        @inbounds update_psi_dot[i] += p.Delta_tau*0.5*(s.psi_dot_dot[i] + s.next_psi_dot_dot[i])
    end

    # swap current and next arrays
    (s.psi_dot_dot, s.next_psi_dot_dot) = (s.next_psi_dot_dot, s.psi_dot_dot)

end

