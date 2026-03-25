from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS (MATCH PAPER)
# ==========================================================
H = 1.0
Lx = 50 * H
Ly = 8 * H

Nx, Ny = 200, 60
dx, dy = Lx / Nx, Ly / Ny

Re = 100
nu = 1.0 / Re

dt = 0.002
nt = 2000

# epsilon (choose: 0, H, 2H)
eps = H

# ==========================================================
# FIELDS
# ==========================================================
u = np.ones((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# ==========================================================
# OBSTACLE (CORRECT BLOCK-ON-BLOCK GEOMETRY)
# ==========================================================

mask = np.zeros((Ny, Nx), dtype=bool)

# define base size using domain height
H_grid = Ny // 8

# -------------------------
# Bottom block (base)
# -------------------------
base_width  = 3 * H_grid
base_height = 1 * H_grid

# -------------------------
# Top block
# -------------------------
top_width  = 1.5 * H_grid
top_height = 1 * H_grid

# center of domain
center_y = Ny // 2

# -------------------------
# Vertical placement
# -------------------------
base_y1 = center_y - base_height//2
base_y2 = base_y1 + base_height

top_y1 = base_y2
top_y2 = top_y1 + top_height

# -------------------------
# Horizontal placement
# -------------------------
center_x = Nx // 2

base_x1 = center_x - base_width//2
base_x2 = base_x1 + base_width

top_x1 = center_x - int(top_width//2)
top_x2 = top_x1 + int(top_width)

# -------------------------
# Apply mask
# -------------------------
mask[base_y1:base_y2, base_x1:base_x2] = True
mask[top_y1:top_y2, top_x1:top_x2] = True

# ==========================================================
# PETSc MATRIX (PRESSURE POISSON)
# ==========================================================
N = Nx * Ny
A = PETSc.Mat().createAIJ([N, N])

def idx(i, j): return i * Nx + j

for i in range(Ny):
    for j in range(Nx):
        r = idx(i, j)

        if 1 <= i < Ny-1 and 1 <= j < Nx-1:
            A[r, r] = -2/dx**2 - 2/dy**2
            A[r, idx(i, j+1)] = 1/dx**2
            A[r, idx(i, j-1)] = 1/dx**2
            A[r, idx(i+1, j)] = 1/dy**2
            A[r, idx(i-1, j)] = 1/dy**2
        else:
            A[r, r] = 1.0

        # outlet pressure BC (VERY IMPORTANT)
        if j == Nx-1:
            A[r, r] = 1.0

A.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('gmres')
ksp.getPC().setType('ilu')

# ==========================================================
# TIME LOOP
# ==========================================================
for n in range(nt):

    un = u.copy()
    vn = v.copy()

    # ============================
    # ADVECTION + DIFFUSION (UPWIND)
    # ============================
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):

            if mask[i, j]:
                continue

            dudx = (un[i,j]-un[i,j-1])/dx if un[i,j]>0 else (un[i,j+1]-un[i,j])/dx
            dudy = (un[i,j]-un[i-1,j])/dy if vn[i,j]>0 else (un[i+1,j]-un[i,j])/dy

            dvdx = (vn[i,j]-vn[i,j-1])/dx if un[i,j]>0 else (vn[i,j+1]-vn[i,j])/dx
            dvdy = (vn[i,j]-vn[i-1,j])/dy if vn[i,j]>0 else (vn[i+1,j]-vn[i,j])/dy

            u[i,j] = un[i,j] - dt*(un[i,j]*dudx + vn[i,j]*dudy) \
                     + nu*dt*((un[i,j+1]-2*un[i,j]+un[i,j-1])/dx**2 +
                              (un[i+1,j]-2*un[i,j]+un[i-1,j])/dy**2)

            v[i,j] = vn[i,j] - dt*(un[i,j]*dvdx + vn[i,j]*dvdy) \
                     + nu*dt*((vn[i,j+1]-2*vn[i,j]+vn[i,j-1])/dx**2 +
                              (vn[i+1,j]-2*vn[i,j]+vn[i-1,j])/dy**2)

    # ============================
    # BOUNDARY CONDITIONS
    # ============================

    # Inlet
    u[:, 0] = 1
    v[:, 0] = 0

    # Walls
    v[0, :] = 0
    v[-1, :] = 0
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]

    # Outlet (convective approx)
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]

    # Obstacle
    u[mask] = 0
    v[mask] = 0

    # ============================
    # PRESSURE RHS
    # ============================
    b = PETSc.Vec().createSeq(N)

    for i in range(1, Ny-1):
        for j in range(1, Nx-1):

            if mask[i, j]:
                continue

            val = ((u[i,j+1]-u[i,j-1])/(2*dx) +
                   (v[i+1,j]-v[i-1,j])/(2*dy)) / dt

            b[idx(i,j)] = val

    b.assemble()

    x = PETSc.Vec().createSeq(N)
    ksp.solve(b, x)

    for i in range(Ny):
        for j in range(Nx):
            p[i,j] = x[idx(i,j)]

    # enforce outlet pressure
    p[:, -1] = 0

    # ============================
    # VELOCITY CORRECTION
    # ============================
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):

            if mask[i, j]:
                continue

            u[i,j] -= dt*(p[i,j+1]-p[i,j-1])/(2*dx)
            v[i,j] -= dt*(p[i+1,j]-p[i-1,j])/(2*dy)
            
            u[mask] = 0
            v[mask] = 0

# ==========================================================
# VISUALIZATION (ZOOMED + STREAMLINES)
# ==========================================================

plt.imshow(mask, origin='lower', cmap='gray')
plt.title("Body check")
plt.savefig("body.png")

X, Y = np.meshgrid(np.linspace(0,Lx,Nx), np.linspace(0,Ly,Ny))

plt.figure(figsize=(12,3))

plt.contourf(X, Y, u,-1, levels=60)
plt.colorbar()

# overlay body
plt.contour(X, Y, mask, levels=[0.5], colors='black')

# plt.streamplot(X, Y, u, v, density=2, color='k')

plt.xlim(15, 35)
plt.ylim(0, 8)

plt.title("Flow around obstacle")
plt.savefig("Petsc.png")
plt.close()
