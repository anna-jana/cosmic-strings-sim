import sympy as sp

# power series approach doesnt seem to work :(
# the constant term is always = 0 hence we cant fullfill the bc at x = 0 f = 1
print("power series approach for x -> 0")

n_max = 6
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
