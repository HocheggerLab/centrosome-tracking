import pint

u = pint.UnitRegistry()
P = u.gram / u.centimeter / u.second
SIvis = u.pascal * u.second
CytoV = u.newton * 1e-12 * u.second / (u.meter * 1e-6) ** 2

q1 = 50 * P
print(P, SIvis)
print(q1)
print(q1.to(SIvis))
print(q1.to(CytoV))
