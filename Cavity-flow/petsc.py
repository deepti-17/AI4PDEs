import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================================
# PARAMETERS
# ==========================================================
nx = 41
ny = 41
nt = 1500
nit = 100

L = 1.0
dx = L / (nx - 1)
dy = L / (ny - 1)

x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

rho = 1.0
nu = 0.01
dt = 0.001

# Reynolds number
Re = 1.0 / nu

print("===================================")
print("Lid Driven Cavity Flow Simulation")
print("Grid:", nx, "x", ny)
print("Time steps:", nt)
print("dt:", dt)
print("Viscosity (nu):", nu)
print("Reynolds Number:", Re)
print("===================================")

# ==========================================================
# INITIALIZE
# ==========================================================
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# ==========================================================
# BUILD RHS
# ==========================================================
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
        ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
         (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
        ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
        2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
             (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
        ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

# ==========================================================
# PRESSURE POISSON
# ==========================================================
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)

    for _ in range(nit):
        pn = p.copy()

        p[1:-1, 1:-1] = (
            ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
             (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
            (2 * (dx**2 + dy**2)) -
            dx**2 * dy**2 /
            (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        )

        # EXACT BCs
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]
        p[0, :] = p[1, :]
        p[-1, :] = 0

    return p

# ==========================================================
# MAIN SOLVER
# ==========================================================
def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(np.zeros_like(p), rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        for i in range(1, ny-1):
            for j in range(1, nx-1):

                if un[i,j] > 0:
                    du_dx = (un[i,j] - un[i,j-1]) / dx
                else:
                    du_dx = (un[i,j+1] - un[i,j]) / dx

                if vn[i,j] > 0:
                    du_dy = (un[i,j] - un[i-1,j]) / dy
                else:
                    du_dy = (un[i+1,j] - un[i,j]) / dy

                if un[i,j] > 0:
                    dv_dx = (vn[i,j] - vn[i,j-1]) / dx
                else:
                    dv_dx = (vn[i,j+1] - vn[i,j]) / dx

                if vn[i,j] > 0:
                    dv_dy = (vn[i,j] - vn[i-1,j]) / dy
                else:
                    dv_dy = (vn[i+1,j] - vn[i,j]) / dy

                u[i,j] = (un[i,j]
                          - un[i,j]*dt*du_dx
                          - vn[i,j]*dt*du_dy
                          - dt/(rho*2*dx)*(p[i,j+1] - p[i,j-1])
                          + nu*(dt/dx**2*(un[i,j+1] - 2*un[i,j] + un[i,j-1]) +
                                dt/dy**2*(un[i+1,j] - 2*un[i,j] + un[i-1,j])))

                v[i,j] = (vn[i,j]
                          - un[i,j]*dt*dv_dx
                          - vn[i,j]*dt*dv_dy
                          - dt/(rho*2*dy)*(p[i+1,j] - p[i-1,j])
                          + nu*(dt/dx**2*(vn[i,j+1] - 2*vn[i,j] + vn[i,j-1]) +
                                dt/dy**2*(vn[i+1,j] - 2*vn[i,j] + vn[i-1,j])))

        # BCs
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1

        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    return u, v, p

# ==========================================================
# RUN
# ==========================================================
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

# ==========================================================
# PLOTTING
# ==========================================================
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8,6))

# Pressure contour
plt.contourf(X, Y, p, levels=20, cmap='rainbow')
plt.colorbar(label="Pressure")

# Pressure lines
plt.contour(X, Y, p, colors='black', linewidths=0.5)

# Velocity vectors (quiver)
plt.quiver(X[::2, ::2], Y[::2, ::2],
           u[::2, ::2], v[::2, ::2],
           scale=5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pressure and Velocity fields')

plt.savefig("quiver_plot.png")
plt.show()

print("✅ Plot saved as: quiver_plot.png")


fig = plt.figure(figsize=(11, 7), dpi=100)

plt.contourf(X, Y, p, alpha=0.5, cmap=cm.coolwarm)
plt.colorbar()

# Optional contour lines (keep commented to match your image)
# plt.contour(X, Y, p, cmap=cm.coolwarm)

plt.streamplot(X, Y, u, v, density=1, linewidth=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Streamlines of Cavity Flow')

plt.savefig("streamplot.png")
plt.show()

print("✅ Streamplot saved as streamplot.png")

