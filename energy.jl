function compute_energy(s :: State, p :: Parameter)
    a = tau_to_a(s.tau)
    theta = angle.(s.phi)
    radial = @. sqrt(2) * abs(s.phi) / a - 1

    mean_axion_kinetic = 0.0
    mean_axion_gradient = 0.0
    mean_radial_kinetic = 0.0
    mean_radial_gradient = 0.0
    mean_radial_potential = 0.0
    mean_interaction = 0.0

    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds @simd for ix in 1:p.N
                R = real(s.phi[ix, iy, iz])
                I = imag(s.phi[ix, iy, iz])
                R_dot = real(s.phi_dot[ix, iy, iz])
                I_dot = imag(s.phi_dot[ix, iy, iz])

                # axion
                # kinetic
                d_theta_d_tau = (I_dot * R - I * R_dot) / (R^2 - I^2)
                axion_kinetic = 0.5 / a^2 * d_theta_d_tau^2
                # gradient
                diff_theta_x = theta[mod1(ix + 1, p.N), iy, iz] - theta[mod1(ix - 1, p.N), iy, iz]
                diff_theta_y = theta[ix, mod1(iy + 1, p.N), iz] - theta[ix, mod1(iy - 1, p.N), iz]
                diff_theta_z = theta[ix, iy, mod1(iz - 1, p.N)] - theta[ix, iy, mod1(iz - 1, p.N)]
                axion_gradient = 0.5 / p.dx^2 * (diff_theta_x^2 + diff_theta_y^2 + diff_theta_z^2)

                # radial mode
                # kinetic
                d_r_d_tau = (R * R_dot + I * I_dot) / abs(s.phi[ix, iy, iz])
                radial_kinetic = 0.5 / a^2 * d_r_d_tau^2
                # gradient
                diff_radial_x = radial[mod1(ix + 1, p.N), iy, iz] - radial[mod1(ix - 1, p.N), iy, iz]
                diff_radial_y = radial[ix, mod1(iy + 1, p.N), iz] - radial[ix, mod1(iy - 1, p.N), iz]
                diff_radial_z = radial[ix, iy, mod1(iz - 1, p.N)] - radial[ix, iy, mod1(iz - 1, p.N)]
                radial_gradient = 0.5 / p.dx^2 * (diff_radial_x^2 + diff_radial_y^2 + diff_radial_z^2)

                # potential
                inner = radial[ix, iy, iz]^2 - 2.0*radial[ix, iy, iz]
                radial_potential = inner^2 / 8.0

                # interaction
                interaction = inner * (axion_kinetic + axion_gradient)

                mean_axion_kinetic += axion_kinetic
                mean_axion_gradient += axion_gradient
                mean_radial_kinetic += radial_kinetic
                mean_radial_gradient += radial_gradient
                mean_radial_potential += radial_potential
                mean_interaction += interaction
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

