Base.@kwdef mutable struct SingleNodeState <: AbstractState
    tau::Float64
    step::Int
    psi::Array{Complex{Float64}, 3}
    psi_dot::Array{Complex{Float64}, 3}
    psi_dot_dot::Array{Complex{Float64}, 3}
    next_psi_dot_dot::Array{Complex{Float64}, 3}
end

function compute_force!(out::Array{Complex{Float64},3},
                        s::SingleNodeState, p::Parameter)
    a = tau_to_a(s.tau)
    Threads.@threads for iz in 1:p.N
        for iy in 1:p.N
            @simd for ix in 1:p.N
                @inbounds begin
                    psi = s.psi[ix, iy, iz]
                    left = s.psi[mod1(ix + 1, p.N), iy, iz]
                    right = s.psi[mod1(ix - 1, p.N), iy, iz]
                    front = s.psi[ix, mod1(iy + 1, p.N), iz]
                    back = s.psi[ix, mod1(iy - 1, p.N), iz]
                    top = s.psi[ix, iy, mod1(iz + 1, p.N)]
                    bottom = s.psi[ix, iy, mod1(iz - 1, p.N)]
                    out[ix, iy, iz] = force_stecil(
                        psi, left, right, front, back, top, bottom, a, p)
                end
            end
        end
    end
end

function random_field_single_node(p::Parameter)
    hat = Array{Float64,3}(undef, (p.N, p.N, p.N))
    ks = FFTW.fftfreq(p.N, 1 / p.dx) .* (2 * pi)
    @inbounds for iz in 1:p.N
        @inbounds for iy in 1:p.N
            @inbounds @simd for ix in 1:p.N
                kx = ks[ix]
                ky = ks[iy]
                kz = ks[iz]
                k = sqrt(kx^2 + ky^2 + kz^2)
                hat[ix, iy, iz] = k <= p.k_max ? (rand() * 2 - 1) * field_max : 0.0
            end
        end
    end
    field = FFTW.ifft(hat)
    return field ./ mean(abs.(field))
end

function SingleNodeState(p::Parameter)
    s = SingleNodeState(
        tau=p.tau_start,
        step=0,
        psi=random_field_single_node(p),
        psi_dot=random_field_single_node(p),
        psi_dot_dot=Array{Float64,3}(undef, (p.N, p.N, p.N)),
        next_psi_dot_dot=Array{Float64,3}(undef, (p.N, p.N, p.N)),
    )

    compute_force!(s.psi_dot_dot, s, p)

    return s
end

@inline function get_update_domain(__s::SingleNodeState, A::Array{Complex{Float64}, 3})
    return A
end

