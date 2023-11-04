@inline function compute_theta_dot(a :: Float64, psi :: Complex{Float64}, psi_dot :: Complex{Float64})
    R = real(psi)
    I = imag(psi)
    R_dot = real(psi_dot)
    I_dot = imag(psi_dot)
    d_theta_d_tau = (I_dot * R - I * R_dot) / (R^2 - I^2)
    return d_theta_d_tau / a
end

@inline function calc_k_max_grid(n, d)
    if n % 2 == 0
        return 2 * pi * (n / 2) / (d*n)
    else
        return 2 * pi * ((n - 1) / 2) / (d*n)
    end
end

@inline function idx_to_k(p, idx)
    return idx < div(p.N, 2) ? idx : -div(p.N, 2) + idx  - div(p.N, 2)
end

@inline function substract_wave_numbers(p, i, j)
    k = idx_to_k(p, i)
    k_prime = idx_to_k(p, j)
    k_diff = mod(k - k_prime + div(p.N, 2), p.N) - div(p.N, 2)
    idx_diff = k_diff < 0 ? k_diff + p.N : k_diff
    return idx_diff
end

function power_spectrum_utils(p::Parameter, a)
    # prepare histogram
    dx_physical = p.dx * a
    kmax = calc_k_max_grid(p.N, dx_physical)
    physical_ks = AbstractFFTs.fftfreq(p.N, 1 / dx_physical) * 2*pi

    Delta_k = 2*pi / (p.N * dx_physical)
    bin_width = kmax / p.nbins

    surface_element = map(1:p.nbins) do i
        vol = 4.0/3.0 * pi * (((i + 1)*bin_width)^3 - (i*bin_width)^3)
        area = 4*pi * (i*bin_width + bin_width/2)^2
        area / vol * Delta_k^3
    end

    bin_ks = [i * bin_width + bin_width/2 for i in 1:p.nbins]

    return physical_ks, bin_width, surface_element, bin_ks
end

function compute_integration_spheres(p::Parameter, physical_ks, bin_width)
    spheres = [Tuple{Int, Int, Int}[] for _ in 1:p.nbins]
    for i in 1:p.nbins
        bin_k_min = i * bin_width
        bin_k_max = bin_k_min + bin_width
        @inbounds for iz in 1:p.N
            @inbounds for iy in 1:p.N
                @inbounds for ix in 1:div(p.N, 2) + 1
                    k2 = physical_ks[ix]^2 + physical_ks[iy]^2 + physical_ks[iz]^2
                    if k2 >= bin_k_min^2 && k2 <= bin_k_max^2
                        push!(spheres[i], (ix, iy, iz))
                    end
                end
            end
        end
    end

end


function compute_power_spectrum(p::Parameter, field, spheres, surface_element, bin_ks)
    field_fft = FFTW.rfft(field)
    # P_field(k) = k^2 / L^3 \int d \Omega / 4\pi 0.5 * | field(k) |^2
    spectrum = zeros(p.nbins)
    for i in 1:p.nbins
        for (ix, iy, iz) in spheres[i]
            spectrum[i] += abs2(field_fft[ix, iy, iz])
        end
        spectrum[i] *= surface_element[i]
        spectrum[i] *= bin_ks[i]^2 / p.L^3 / (4 * pi) * 0.5
    end

end

# compute PPSE (pseudo-power-spectrum-estimator) of the theta-dot field
# -> the spectrum of number denseties of axtion
function compute_spectrum_ppse(p :: Parameter, s :: SingleNodeState, strings :: Vector{Vector{SVector{3, Float64}}})
    a = tau_to_a(s.tau)
    theta_dot = compute_theta_dot.(a, s.psi, s.psi_dot)

    # compute W (string mask)
    W = ones(p.N, p.N, p.N)
    for string in strings
        for point in string
            @inbounds for x_offset in -p.radius:p.radius
                @inbounds for y_offset in -p.radius:p.radius
                    @inbounds @simd for z_offset in -p.radius:p.radius
                        r2 = x_offset^2 + y_offset^2 + z_offset^2
                        if r2 <= p.radius^2
                            ix = mod1(floor(Int, point[1] + 1.0), p.N)
                            iy = mod1(floor(Int, point[2] + 1.0), p.N)
                            iz = mod1(floor(Int, point[3] + 1.0), p.N)
                            W[ix, iy, iz] = zero(eltype(W))
                        end
                    end
                end
            end
        end
    end

    # mask out the strings in theta dot
    theta_dot .*= W

    physical_ks, bin_width, surface_element, bin_ks = power_spectrum_utils(p, a)
    spheres = compute_integration_spheres(p, physical_ks, bin_width)

    spectrum_uncorrected = compute_power_spectrum(p, theta_dot, spheres, surface_element, bin_ks)

    W_fft = FFTW.rfft(W)

    # compute M
    # M = 1 / (L^3)^2 * \int d \Omega / 4\pi d \Omega' / 4\pi |W(\vec{k} - \vec{k}')|^2
    # NOTE: this is the computationally most expensive part!
    M = zeros(p.nbins, p.nbins)
    f = p.L^6 * (4 * pi)^2
    for i in 1:p.nbins
        for j in i:p.nbins
            println("$((i, j)) of $((p.nbins, p.nbins))")
            # integrate spheres
            s_atomic = Threads.Atomic{Float64}(0.0)
            Threads.@threads for idx1 in spheres[i]
                for idx2 in spheres[j]
                    ix = substract_wave_numbers(p, idx1[1] - 1, idx2[1] - 1)
                    if ix >= size(W_fft, 1)
                        ix = ix - size(W_fft, 1)
                    end
                    iy = substract_wave_numbers(p, idx1[2] - 1, idx2[2] - 1)
                    iz = substract_wave_numbers(p, idx1[3] - 1, idx2[3] - 1)
                    Threads.atomic_add!(s_atomic, abs2(@inbounds W_fft[ix + 1, iy + 1, iz + 1]))
                end
            end
            s = s_atomic[]
            s *= surface_element[i] * surface_element[j] / f
            M[i, j] = M[j, i] = s
        end
    end

    M_inv = inv(M)

    for j in 1:p.nbins
        for i in 1:p.nbins
            M_inv[i, j] = M_inv[i, j] * (2 * pi)^2 / (bin_width * bin_ks[i]^2 * bin_ks[j]^2)
        end
    end

    spectrum_corrected = M_inv * spectrum_uncorrected # NOTE: matrix multiply!

    for i in p.nbins
        spectrum_corrected[i] *= bin_ks[i]^2 / p.L^3 / (2*pi^2) * bin_width
    end

    return bin_ks, spectrum_corrected
end

function compute_spectrum_autoscreen(p :: Parameter, s :: SingleNodeState)
    a = tau_to_a(s.tau)
    theta_dot = compute_theta_dot.(a, s.psi, s.psi_dot)
    r = compute_radial_mode.(s, a)

    physical_ks, bin_width, surface_element, bin_ks = power_spectrum_utils(p, a)
    spheres = compute_integration_spheres(p, physical_ks, bin_width)

    screened_theta_dot = @. (1 - r) * theta_dot

    return compute_power_spectrum(p, screened_theta_dot, spheres, surface_element, bin_ks)
end

# all(isapprox.(rfft(theat_dot), fft(theat_dot)[1:div(p.N, 2)+1, :, :]))


function compute_spectrum_autoscreen(p :: Parameter, s :: MPIState)
    a = tau_to_a(s.tau)
    theta_dot = compute_theta_dot.(a, @view s.psi[2:end-1, 2:end-1, 2:end-1], @view s.psi_dot[2:end-1, 2:end-1, 2:end-1])
    r = compute_radial_mode.(s, a)
    screened_theta_dot = @. (1 - r) * theta_dot

    physical_ks, bin_width, surface_element, bin_ks = power_spectrum_utils(p, a)

    # rfft of theta_dot
    transform = PencilFFTs.Transforms.RFFT()
    plan = PencilFFTPlan(s.pen, transform)
    screended_theta_dot_fft = allocate_output(plan)
    screened_theta_dot_pen = PencilArray(s.pen, screened_theta_dot)
    mul!(screended_theta_dot_fft, plan, screened_theta_dot_pen)

    # collect the parts of the integration for our node
    my_range = range_local(screended_theta_dot_fft)
    my_sphere_parts = [Tuple{Int, Int, Int}[] for _ in 1:p.nbins]
    for i in 1:p.nbins
        bin_k_min = i * bin_width
        bin_k_max = bin_k_min + bin_width
        for iz in my_range[3]
            for iy in my_range[2]
                for ix in my_range[1]
                    @inbounds k2 = physical_ks[ix]^2 + physical_ks[iy]^2 + physical_ks[iz]^2
                    if k2 >= bin_k_min^2 && k2 <= bin_k_max^2
                        @inbounds push!(my_sphere_parts[i], (ix, iy, iz))
                    end
                end
            end
        end
    end

    screened_theta_dot_fft_global = global_view(screended_theta_dot_fft)
    # P_field(k) = k^2 / L^3 \int d \Omega / 4\pi 0.5 * | field(k) |^2
    spectrum = zeros(p.nbins)
    for i in 1:p.nbins
        for (ix, iy, iz) in my_sphere_parts[i]
            spectrum[i] += abs2(screened_theta_dot_fft_global[ix, iy, iz])
        end

        MPI.Reduce(spectrum[i], +, s.root, s.comm)

        if s.rank == s.root
            spectrum[i] *= surface_element[i]
            spectrum[i] *= bin_ks[i]^2 / p.L^3 / (4 * pi) * 0.5
        end
    end

    if s.rank == s.root
        return bin_ks, spectrum
    else
        return nothing, nothing
    end
end

# compute the instanteous emission spectrum defined in the paper by gorghetto (axion strings: the attractive solution, eq. 33)
function compute_instanteous_emission_spectrum(P1, P2, k1, k2, tau1, tau2)
    # re interpolate such that P1 and P2 have a shared support
    k_min = max(k1[1], k2[1])
    k_max = min(k1[end], k2[end])
    ks = range(k_min, k_max, length=length(k1))

    P1_interp = linear_interpolation(k1, P1)(ks)
    P2_interp = linear_interpolation(k2, P2)(ks)

    # compute time scales
    t1 = tau_to_t(tau1)
    t2 = tau_to_t(tau2)
    a1 = tau_to_a(tau1)
    a2 = tau_to_a(tau2)
    t_mid = (t2 + t1) / 2
    a_mid = t_to_a(t_mid)
    log_mid = tau_to_log(t_to_tau(t_mid))

    # finite difference
    F = @. (a2^3 * P2_interp - a1^3 * P1_interp) / (t2 - t1) / a_mid^3

    # normalize
    A = sum(@. (F[2:end] + F[1:end-1]) / 2 * (ks[2:end] + ks[1:end-1]) / 2)
    F ./= A

    return log_mid, ks, F
end

