function init_state_single_node(p::Parameter)
   s = State(
        tau=p.tau_start,
        step=0,
        psi=random_field(p),
        psi_dot=random_field(p),
        psi_dot_dot=new_field_array(p),
        next_psi_dot_dot=new_field_array(p),
   )

   compute_force_single_node!(s.psi_dot_dot, s, p)

   return s
end

function compute_force_single_node!(out :: Array{Complex{Float64}, 3},
                                    s :: State, p :: Parameter)
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

