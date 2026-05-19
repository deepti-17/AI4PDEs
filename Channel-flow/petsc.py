from petsc4py import PETSc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Running Navier–Stokes Channel Flow (PETSc)...")

# ==========================================================
# PARAMETERS
# ==========================================================
nx, ny = 41, 41
nt = 10000
nit = 50

Lx, Ly = 2.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

rho = 1.0
nu = 0.1
dt = 0.001
F = 1.0

# ==========================================================
# GRID
# ==========================================================
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# ==========================================================
# INITIALIZE
# ==========================================================
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# ==========================================================
# PETSc POISSON MATRIX (PERIODIC X)
# ==========================================================
N = nx * ny

def idx(i, j):
    return i * nx + j

A = PETSc.Mat().create()
A.setSizes([N, N])
A.setType('aij')
A.setUp()

for i in range(ny):
    for j in range(nx):
        row = idx(i, j)

        if i == 0 or i == ny-1:
            A[row, row] = 1.0
        else:
            jp = (j + 1) % nx
            jm = (j - 1) % nx

            A[row, idx(i,j)] = -2*(1/dx**2 + 1/dy**2)
            A[row, idx(i,jp)] = 1/dx**2
            A[row, idx(i,jm)] = 1/dx**2
            A[row, idx(i+1,j)] = 1/dy**2
            A[row, idx(i-1,j)] = 1/dy**2

A.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
ksp.getPC().setType('jacobi')

nullspace = PETSc.NullSpace().create(constant=True)
A.setNullSpace(nullspace)

# ==========================================================
# TIME LOOP
# ==========================================================
for n in range(nt):

    un = u.copy()
    vn = v.copy()

    # RHS (b)
    b = np.zeros((ny, nx))
    b[1:-1,1:-1] = rho * (
        (1/dt)*((un[1:-1,2:] - un[1:-1,:-2])/(2*dx) +
                (vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy))
        - ((un[1:-1,2:] - un[1:-1,:-2])/(2*dx))**2
        - 2*((un[2:,1:-1] - un[:-2,1:-1])/(2*dy) *
             (vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx))
        - ((vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy))**2
    )

    # PETSc solve
    b_vec = PETSc.Vec().createSeq(N)
    p_vec = PETSc.Vec().createSeq(N)

    for i in range(ny):
        for j in range(nx):
            b_vec[idx(i,j)] = b[i,j]

    ksp.solve(b_vec, p_vec)

    for i in range(ny):
        for j in range(nx):
            p[i,j] = p_vec[idx(i,j)]

    p -= np.mean(p)

    # Velocity update (central difference)
    u[1:-1,1:-1] = (
        un[1:-1,1:-1]
        - un[1:-1,1:-1]*dt*(un[1:-1,2:] - un[1:-1,:-2])/(2*dx)
        - vn[1:-1,1:-1]*dt*(un[2:,1:-1] - un[:-2,1:-1])/(2*dy)
        - dt/rho*(p[1:-1,2:] - p[1:-1,:-2])/(2*dx)
        + nu*(dt/dx**2*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) +
              dt/dy**2*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]))
        + F*dt
    )

    v[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - un[1:-1,1:-1]*dt*(vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx)
        - vn[1:-1,1:-1]*dt*(vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy)
        - dt/rho*(p[2:,1:-1] - p[:-2,1:-1])/(2*dy)
        + nu*(dt/dx**2*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) +
              dt/dy**2*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]))
    )

    # BCs
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]
    v[:,0] = v[:,-2]
    v[:,-1] = v[:,1]

    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0

    if n % 200 == 0:
        print(f"Step {n}")

# ==========================================================
# PLOTS
# ==========================================================
plt.figure(figsize=(11,7))
plt.quiver(X[::3,::3], Y[::3,::3], u[::3,::3], v[::3,::3])
plt.title("Sparse Quiver")
plt.savefig("Sparse.png")
plt.close()

plt.figure(figsize=(11,7))
plt.quiver(X, Y, u, v)
plt.title("Dense Quiver")
plt.savefig("Dense.png")
plt.close()

# Analytical comparison
H = Ly
u_ana = (F/(2*nu))*y*(H-y)
u_cfd = u[:, int(nx/2)]

plt.figure(figsize=(10,8))
plt.plot(u_ana, y, 'o', label='Analytical')
plt.plot(u_cfd, y, label='CFD')

plt.legend()
plt.grid()
plt.gca().invert_yaxis()
plt.title("Comparison")
plt.savefig("Comparison.png")
plt.close()

print("✅ DONE — matches webpage method")
