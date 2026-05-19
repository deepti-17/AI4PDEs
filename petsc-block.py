from petsc4py import PETSc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Running Navier–Stokes Channel Flow with Solid Block (PETSc)...")

# ==========================================================
# PARAMETERS
# ==========================================================
nx, ny = 41, 41
nt = 10000

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
# SOLID BLOCK (OBSTACLE)
# ==========================================================
solid = np.zeros((ny, nx))

# Smaller centered block
bx_center = nx // 2
by_center = ny // 2

block_width = nx // 10     # smaller width
block_height = ny // 6     # smaller height

bx_start = bx_center - block_width // 2
bx_end   = bx_center + block_width // 2

by_start = by_center - block_height // 2
by_end   = by_center + block_height // 2

solid[by_start:by_end, bx_start:bx_end] = 1

# ==========================================================
# PETSc POISSON MATRIX (WITH BLOCK + PERIODIC X)
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

        if i == 0 or i == ny-1 or solid[i, j] == 1:
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

    # ======================================================
    # RHS (Pressure Poisson)
    # ======================================================
    b = np.zeros((ny, nx))

    b[1:-1,1:-1] = rho * (
        (1/dt)*((un[1:-1,2:] - un[1:-1,:-2])/(2*dx) +
                (vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy))
        - ((un[1:-1,2:] - un[1:-1,:-2])/(2*dx))**2
        - 2*((un[2:,1:-1] - un[:-2,1:-1])/(2*dy) *
             (vn[1:-1,2:] - vn[1:-1,:-2])/(2*dx))
        - ((vn[2:,1:-1] - vn[:-2,1:-1])/(2*dy))**2
    )

    # Zero inside solid
    b[solid == 1] = 0

    # ======================================================
    # PETSc SOLVE
    # ======================================================
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

    # ======================================================
    # VELOCITY UPDATE
    # ======================================================
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

    # ======================================================
    # BOUNDARY CONDITIONS
    # ======================================================
    # Periodic X
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]
    v[:,0] = v[:,-2]
    v[:,-1] = v[:,1]

    # Walls
    u[0,:] = 0
    u[-1,:] = 0
    v[0,:] = 0
    v[-1,:] = 0

    # Solid block (no-slip)
    u[solid == 1] = 0
    v[solid == 1] = 0

    if n % 500 == 0:
        print(f"Step {n}")

# ==========================================================
# PLOTTING
# ==========================================================
plt.figure(figsize=(11,7))
plt.contourf(X, Y, solid, levels=[0.5,1], colors='black', alpha=0.3)
plt.quiver(X[::3,::3], Y[::3,::3], u[::3,::3], v[::3,::3])
plt.title("Flow over Solid Block")
plt.savefig("Sparse_block.png")
plt.close()

plt.figure(figsize=(11,7))
plt.quiver(X, Y, u, v)
plt.title("Dense Flow")
plt.savefig("Dense_block.png")
plt.close()

print("✅ DONE — Flow over block simulation complete")

# ==========================================================
# CONTOUR PLOT
# ==========================================================

# Velocity magnitude
vel_mag = np.sqrt(u**2 + v**2)

plt.figure(figsize=(10,7))
plt.contourf(X, Y, u, 20)
plt.colorbar(label='u-velocity')
plt.contour(X, Y, solid, levels=[0.5], colors='black')
plt.title("Contour of U-Velocity")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Contour_u.png")
plt.close()


plt.figure(figsize=(10,7))
# Mask solid region for clean plot
v_masked = np.ma.masked_where(solid==1, v)
plt.contourf(X, Y, v_masked, 20)
plt.colorbar(label='v-velocity')
# Draw block boundary
plt.contour(X, Y, solid, levels=[0.5], colors='black')
plt.title("Contour of V-Velocity")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Contour_v.png")
plt.close()

