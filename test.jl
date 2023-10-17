include("AxionStrings.jl")
include("util.jl")

s, p = AxionStrings.init()
AxionStrings.run_simulation!(s, p)
strs = AxionStrings.detect_strings(s, p)
spec = wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
save_spectrum(spec, "rfft_spec.hdf5")
# spec = wavenumber, power = load_spectrum("spec.hdf5")
