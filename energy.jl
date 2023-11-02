function compute_energy(s :: AbstractState, p :: Parameter)
    a = tau_to_a(s.tau)
    H = t_to_H(tau_to_t(s.tau))
    theta = angle.(s.psi)
    radial = @. sqrt(2) * abs(s.psi) / a - 1

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
                    R = real(s.psi[ix, iy, iz])
                    I = imag(s.psi[ix, iy, iz])
                    R_dot = real(s.psi_dot[ix, iy, iz])
                    I_dot = imag(s.psi_dot[ix, iy, iz])

                    # axion
                    # kinetic
                    d_theta_d_tau = (I_dot * R - I * R_dot) / abs(s.psi[ix, iy, iz])
                    axion_kinetic = 0.5 / a^2 * d_theta_d_tau^2
                    # gradient
                    diff_theta_x = theta[mod1(ix + 1, p.N), iy, iz] - theta[mod1(ix - 1, p.N), iy, iz]
                    diff_theta_y = theta[ix, mod1(iy + 1, p.N), iz] - theta[ix, mod1(iy - 1, p.N), iz]
                    diff_theta_z = theta[ix, iy, mod1(iz + 1, p.N)] - theta[ix, iy, mod1(iz - 1, p.N)]
                    axion_gradient = 0.5 / (p.dx * 2)^2 * (diff_theta_x^2 + diff_theta_y^2 + diff_theta_z^2) / a^2

                    # radial mode
                    # kinetic
                    d_abs_psi_d_tau = (R * R_dot + I * I_dot) / abs(s.psi[ix, iy, iz])
                    abs_psi = abs(s.psi[ix, iy, iz])
                    d_r_d_tau = sqrt(2) * (d_abs_psi_d_tau / a - H * abs_psi)
                    radial_kinetic = 0.5 / a^2 * d_r_d_tau^2 # 0.5 (dr/dt)^2

                    # gradient
                    diff_radial_x = radial[mod1(ix + 1, p.N), iy, iz] - radial[mod1(ix - 1, p.N), iy, iz]
                    diff_radial_y = radial[ix, mod1(iy + 1, p.N), iz] - radial[ix, mod1(iy - 1, p.N), iz]
                    diff_radial_z = radial[ix, iy, mod1(iz + 1, p.N)] - radial[ix, iy, mod1(iz - 1, p.N)]
                    radial_gradient = 0.5 / (p.dx * 2)^2 * (diff_radial_x^2 + diff_radial_y^2 + diff_radial_z^2) / a^2

                    # potential
                    inner = radial[ix, iy, iz]^2 + 2.0*radial[ix, iy, iz]
                    radial_potential = inner^2 / 8.0

                    # interaction
                    interaction = inner / 2 * (axion_kinetic + axion_gradient)

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

