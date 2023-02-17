# TODO list for cosmic string simulation

[x] generate initial conditions using inverse FFT
[x] propgatation algorithm
[x] plot slices
[x] string detection
[x] string plotting
[~] string length and loop size distribution
[x] shared memory paralism
[ ] distributed memory paralism
[x] compute energy components
[x] make some constants parameters (defines -> global variables) =====> makes the code slower
[ ] compute spectrum (python)
[ ] prefactors between kawasaki and gorghetto differ
[ ] compute instantaeous spectrum (python)
[ ] compute spectrum and instantaeous spectrum (c)
[ ] compute gamme factor for strings
[x] exchange parameters between python and c (either call c code from python or parameter file)
[ ] study creation of initial conditions
[ ] higher order time propagation algorithm
[ ] higher order stecil for laplacian
[ ] GPU code?
[x] output in a newly created directory and python analysis scripts take the directory as an argument
[x] parse cmdline args
[x] reverse loop order for cache efficentcy
[x] const correctness
[ ] use arrays for allocation
[x] document eom
[x] document energy
[ ] document propagation algorithm
[ ] video of strings
[~] I think the detected strings in c and python are not the same
[x] i think we cant go to k_max in the spectrum computation because then we dont integrate a hole sphere (the corners in the grid are cutting the sphere)
