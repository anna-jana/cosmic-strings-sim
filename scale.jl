using PyPlot
using Roots

const Lambda = 400 # [MeV]
const M_pl = 2.435e18 * 1e3 # [MeV]
const g_star = 104.98 # [1]
const alpha = 1.68e-7 # [1]
const alpha0 = 1.46e-3
const n = 6.68 # [1]
const zero_T = sqrt(alpha0) * Lambda^2

function axion_mass_times_f_a(T)
    power_law = sqrt(alpha) * (T / Lambda)^(-n/2) * Lambda^2
    return min(zero_T, power_law)
end

function hubble_at_temperature(T)
    return sqrt(g_star / 90) * pi / M_pl * T^2
end

function solve(f_a)
    function goal(log_T)
        T = exp(log_T)
        return log(axion_mass_times_f_a(T) / f_a / hubble_at_temperature(T))
    end

    return exp(find_zero(goal, log(1e10 * Lambda)))
end

f_a_list = @. 10^(8:12) * 1e3 # [MeV]
T_osc = solve.(f_a_list) # [MeV]
H_osc = hubble_at_temperature.(T_osc) # [MeV]
H_osc_analytic = @. (
        (sqrt(g_star / 90) * pi / M_pl)^(n/(n + 4)) *
        alpha^(2/(n + 4)) *
        f_a_list^(- 4/(n + 4)) *
        Lambda^2
)
logs = @. log(f_a_list / H_osc)
log_analytic = @. log(f_a_list / H_osc_analytic)

figure()
semilogx(f_a_list, logs, label="numerical")
semilogx(f_a_list, log_analytic, ls="--", label="analytical")
xlabel("f_a / MeV")
ylabel("log(f_a / H_osc)")
legend()
savefig("scale.pdf")
show()
