import sympy as sp

############################### for x -> 0 #########################
print("power series approach for x -> 0")

n_max = 10
powers = list(range(n_max + 1))

coeff = sp.symbols(" ".join(["a_" + str(i) for i in powers]))

x = sp.symbols("x")
f = sum([coeff[i] * x**i / sp.factorial(i) for i in powers])
print(f)

lhs = f.diff(x, 2)*x**2 + f.diff(x, 1)*x - f + x**2*f*(1 - f**2)

by_term = [sp.Eq(lhs.diff(x, i).subs(x, 0).simplify(), 0) for i in powers]

params = set(coeff)
subs = []

for i, (c, term) in enumerate(zip(coeff, by_term)):
    print("order:", i)
    if term == True:
        print("any solution")
        continue
    assert term != False
    term = term.subs(subs)
    print(term)
    sols = sp.solve(term, c)
    print(sols)
    assert len(sols) == 1
    subs.append((c, sols[0]))
    params.remove(c)

print("solution:")
f_sol = f.subs(subs)
print(f_sol)
print("parameter:", params)

########################### for x -> inf ####################
print("\n\npower series approch for x -> inf")

n_max = 3
powers = list(range(n_max + 1))
coeff = sp.symbols(" ".join(["b_" + str(i) for i in powers]))
u = sp.symbols("u")
f = sum([coeff[i] * u**i / sp.factorial(i) for i in powers])
print(f)

lhs = u**4 * f.diff(u, 2) + 3 * u**2 * f.diff(u) - u**2 * f + f*(1 - f**2)

by_term = [sp.Eq(lhs.diff(u, i).subs(u, 0).simplify(), 0) for i in powers]

subs = []
for i, (c, term) in enumerate(zip(coeff, by_term)):
    print("order:", i)
    term = term.subs(subs)
    print(term)
    sols = sp.solve(term, c)
    print(sols)
    if i == 0:
        assert 1 in sols
        # we choose the solution 0 bc of the bc that f = 1 for rho -> inf
        subs.append((c, 1))
    else:
        assert len(sols) == 1
        subs.append((c, sols[0]))

f_sol = f.subs(subs)
x = sp.symbols("x")
f_sol = f_sol.subs(u, 1/x)
print("solution:", f_sol)
