function compute_energy_at(
        p::Parameter, a, H,
        psi, psi_dot,
        theta_left, theta_right, theta_front, theta_back, theta_top, theta_bottom,
        radial, radial_left, radial_right, radial_front, radial_back, radial_top, radial_bottom,
    )
    R = real(psi)
    I = imag(psi)
    R_dot = real(psi_dot)
    I_dot = imag(psi_dot)

    ## axion
    # kinetic
    d_theta_d_tau = (I_dot * R - I * R_dot) / abs(psi)
    axion_kinetic = 0.5 / a^2 * d_theta_d_tau^2
    # gradient
    diff_theta_x = theta_left - theta_right
    diff_theta_y = theta_front - theta_back
    diff_theta_z = theta_top - theta_bottom
    axion_gradient = 0.5 / (p.dx * 2)^2 * (diff_theta_x^2 + diff_theta_y^2 + diff_theta_z^2) / a^2

    ## radial mode
    # kinetic
    d_abs_psi_d_tau = (R * R_dot + I * I_dot) / abs(psi)
    abs_psi = abs(psi)
    d_r_d_tau = sqrt(2) * (d_abs_psi_d_tau / a - H * abs_psi)
    radial_kinetic = 0.5 / a^2 * d_r_d_tau^2 # 0.5 (dr/dt)^2

    # gradient
    diff_radial_x = radial_left - radial_right
    diff_radial_y = radial_front - radial_back
    diff_radial_z = radial_top - radial_bottom
    radial_gradient = 0.5 / (p.dx * 2)^2 * (diff_radial_x^2 + diff_radial_y^2 + diff_radial_z^2) / a^2

    # potential
    inner = radial^2 + 2.0 * radial
    radial_potential = inner^2 / 8.0

    ## interaction
    interaction = inner / 2 * (axion_kinetic + axion_gradient)

    return axion_kinetic, axion_gradient, radial_kinetic, radial_gradient, radial_potential, interaction
end

function compute_radial_mode(psi, a)
    # return sqrt(2) * abs(psi) / a - 1
    return sqrt(2) * real(psi / exp(angle(psi) * im)) / a - 1
end

function compute_energy(s::SingleNodeState, p::Parameter)
    a = tau_to_a(s.tau)
    H = t_to_H(tau_to_t(s.tau))
    theta = angle.(s.psi)
    radial = compute_radial_mode.(s.psi, a)

    mean_axion_kinetic = 0.0
    mean_axion_gradient = 0.0
    mean_radial_kinetic = 0.0
    mean_radial_gradient = 0.0
    mean_radial_potential = 0.0
    mean_interaction = 0.0

    Threads.@threads for iz in 1:p.N
        for iy in 1:p.N
            @simd for ix in 1:p.N
                @inbounds begin
                    psi = s.psi[ix, iy, iz]
                    psi_dot = s.psi_dot[ix, iy, iz]

                    theta_left, theta_right = theta[mod1(ix + 1, p.N), iy, iz], theta[mod1(ix - 1, p.N), iy, iz]
                    theta_front, theta_back = theta[ix, mod1(iy + 1, p.N), iz], theta[ix, mod1(iy - 1, p.N), iz]
                    theta_top, theta_bottom = theta[ix, iy, mod1(iz + 1, p.N)], theta[ix, iy, mod1(iz - 1, p.N)]

                    radial_left, radial_right = radial[mod1(ix + 1, p.N), iy, iz], radial[mod1(ix - 1, p.N), iy, iz]
                    radial_front, radial_back = radial[ix, mod1(iy + 1, p.N), iz], radial[ix, mod1(iy - 1, p.N), iz]
                    radial_top, radial_bottom = radial[ix, iy, mod1(iz + 1, p.N)], radial[ix, iy, mod1(iz - 1, p.N)]
                    r = radial[ix, iy, iz]

                    axion_kinetic, axion_gradient, radial_kinetic, radial_gradient, radial_potential, interaction = compute_energy_at(p, a, H,
                        psi, psi_dot,
                        theta_left, theta_right, theta_front, theta_back, theta_top, theta_bottom,
                        r, radial_left, radial_right, radial_front, radial_back, radial_top, radial_bottom)

                    mean_axion_kinetic += axion_kinetic
                    mean_axion_gradient += axion_gradient
                    mean_radial_kinetic += radial_kinetic
                    mean_radial_gradient += radial_gradient
                    mean_radial_potential += radial_potential
                    mean_interaction += interaction
                end
            end
        end
    end

    mean_axion_kinetic /= p.N^3
    mean_axion_gradient /= p.N^3
    mean_radial_kinetic /= p.N^3
    mean_radial_gradient /= p.N^3
    mean_radial_potential /= p.N^3
    mean_interaction /= p.N^3

    mean_axion_total = mean_axion_kinetic + mean_axion_gradient
    mean_radial_total = mean_radial_kinetic + mean_radial_gradient + mean_radial_potential
    mean_total = mean_axion_total + mean_radial_total + mean_interaction

    return (mean_axion_kinetic, mean_axion_gradient, mean_axion_total,
        mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_radial_total,
        mean_interaction, mean_total)
end

function compute_energy(s::MPIState, p::Parameter)
    a = tau_to_a(s.tau)
    H = t_to_H(tau_to_t(s.tau))

    exchange_field!(s)

    theta = angle.(s.psi)
    radial = compute_radial_mode.(s, a)

    mean_axion_kinetic = 0.0
    mean_axion_gradient = 0.0
    mean_radial_kinetic = 0.0
    mean_radial_gradient = 0.0
    mean_radial_potential = 0.0
    mean_interaction = 0.0

    @inbounds for iz in 2:s.lnz
        for iy in 2:s.lny
            @simd for ix in 1:p.N
                psi = s.psi[ix, iy, iz]
                psi_dot = s.psi_dot[ix, iy, iz]

                theta_left, theta_right = theta[ix + 1, iy, iz], theta[ix - 1, iy, iz]
                theta_front, theta_back = theta[ix, iy + 1, iz], theta[ix, iy - 1, iz]
                theta_top, theta_bottom = theta[ix, iy, iz + 1], theta[ix, iy, iz - 1]

                radial_left, radial_right = radial[ix + 1, iy, iz], radial[ix - 1, iy, iz]
                radial_front, radial_back = radial[ix, iy + 1, iz], radial[ix, iy - 1, iz]
                radial_top, radial_bottom = radial[ix, iy, iz + 1], radial[ix, iy, iz - 1]
                r = radial[ix, iy, iz]

                axion_kinetic, axion_gradient, radial_kinetic, radial_gradient, radial_potential, interaction = compute_energy_at(p, a, H,
                    psi, psi_dot,
                    theta_left, theta_right, theta_front, theta_back, theta_top, theta_bottom,
                    r, radial_left, radial_right, radial_front, radial_back, radial_top, radial_bottom)

                mean_axion_kinetic += axion_kinetic
                mean_axion_gradient += axion_gradient
                mean_radial_kinetic += radial_kinetic
                mean_radial_gradient += radial_gradient
                mean_radial_potential += radial_potential
                mean_interaction += interaction
            end
        end
    end

    # sum all the subboxes
    mean_axion_kinetic = MPI.Reduce(mean_axion_kinetic, +, s.root, s.comm)
    mean_axion_gradient = MPI.Reduce(mean_axion_gradient, +, s.root, s.comm)
    mean_radial_kinetic = MPI.Reduce(mean_radial_kinetic, +, s.root, s.comm)
    mean_radial_gradient = MPI.Reduce(mean_radial_gradient, +, s.root, s.comm)
    mean_radial_potential = MPI.Reduce(mean_radial_potential, +, s.root, s.comm)
    mean_interaction = MPI.Reduce(mean_interaction, +, s.root, s.comm)

    mean_axion_kinetic /= p.N^3
    mean_axion_gradient /= p.N^3
    mean_radial_kinetic /= p.N^3
    mean_radial_gradient /= p.N^3
    mean_radial_potential /= p.N^3
    mean_interaction /= p.N^3

    mean_axion_total = mean_axion_kinetic + mean_axion_gradient
    mean_radial_total = mean_radial_kinetic + mean_radial_gradient + mean_radial_potential
    mean_total = mean_axion_total + mean_radial_total + mean_interaction

    return (mean_axion_kinetic, mean_axion_gradient, mean_axion_total,
        mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_radial_total,
        mean_interaction, mean_total)
end

