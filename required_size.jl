include("AxionStrings.jl")
using PyPlot

log_end = range(1.0, 9.0, 100)
required_N = [AxionStrings.sim_params_from_physical_scale(l)[2] for l in log_end]
required_bytes = @. 4 * 4 * required_N^3 / (1024^3)
plot(log_end, required_bytes, label="required by simulation")
axhline(8; color="k", ls="--", label="laptop")
xlabel("scale log(m_r / H)")
ylabel("giga bytes")
yscale("log")
legend()
show()
