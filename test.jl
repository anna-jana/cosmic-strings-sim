include("AxionStrings.jl")
include("util.jl")

s, p = AxionStrings.init()
AxionStrings.run_simulation!(s, p)
strs = AxionStrings.detect_strings(s, p)
# spec = wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
# plot(wavenumber, power)
# save_spectrum(spec, "mc_spec.hdf5")
