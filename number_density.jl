using PyPlot
using QuadGK

function f_gorghetto(l, q)
    return (1 - 1/q) / (1 - (2*q - 1) * exp((1 - q) * l))
end

function calc_f_numerical(l, q)
    function integrant(l)
        return l * exp(l/2) / (1 - exp((1 - q)*l)) * (1 - exp(-q*l))
    end

    I = quadgk(integrant, -5.0, l)[1]
    return 0.5 * (1 - 1 / q) * exp(-0.5*l) / l * I
end

qs = range(0.5, 5.0, length=200)[2:end]
l = 70

figure()
plot(qs, f_gorghetto.(l, qs), label="Gorghettos Formula")
plot(qs, calc_f_numerical.(l, qs), label="numerical integral")
yscale("log")
ylim(1e-4, 1)
xlabel("q")
ylabel(raw"$f(q) = n_a / (8 H \xi \mu / x_0)$")
legend()
title("log = $l")
grid()
savefig("number_density.pdf")
show()
