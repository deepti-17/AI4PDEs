from petsc4py import PETSc
import numpy as np

# ==========================================================
# PARAMETERS
# ==========================================================
Nx, Ny = 5, 5
dx, dy = 1.0/Nx, 1.0/Ny
Re = 100
nu = 1.0 / Re
dt = 0.01
ub = 1.0

# ==========================================================
# INITIAL CONDITIONS
# ==========================================================
u = np.ones((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))

# ==========================================================
# SOLID BODY
# ==========================================================
sigma = np.zeros((Ny, Nx))
sigma[2,2] = 1e8   # match AI4CFD strength

# ==========================================================
# INDEX
# ==========================================================
def idx(i, j): return i * Nx + j
N = Nx * Ny

# ==========================================================
# PRINT MATRIX
# ==========================================================
def print_matrix(A, name):
    dense = np.zeros((N,N))
    for i in range(N):
        cols, vals = A.getRow(i)
        for c,v in zip(cols,vals):
            dense[i,c] = v
    print(f"\n===== {name} =====\n")
    np.set_printoptions(precision=3, suppress=False)
    print(dense)

# ==========================================================
# CORRECT BC (AI4CFD STYLE)
# ==========================================================
def apply_bc(u,v):
    # LEFT & RIGHT → Dirichlet inflow
    u[:,0] = ub
    u[:,-1] = ub

    v[:,0] = 0.0
    v[:,-1] = 0.0

    # TOP & BOTTOM → no-slip
    u[0,:] = 0.0
    u[-1,:] = 0.0
    v[0,:] = 0.0
    v[-1,:] = 0.0

# ==========================================================
# STEP 1: MATRIX FOR u*, v*
# ==========================================================
A = PETSc.Mat().createAIJ([N, N])

for i in range(Ny):
    for j in range(Nx):
        r = idx(i,j)

        if 1 <= i < Ny-1 and 1 <= j < Nx-1:
            A[r,r] = 1 + 2*nu*dt*(1/dx**2 + 1/dy**2)
            A[r, idx(i,j+1)] = -nu*dt/dx**2
            A[r, idx(i,j-1)] = -nu*dt/dx**2
            A[r, idx(i+1,j)] = -nu*dt/dy**2
            A[r, idx(i-1,j)] = -nu*dt/dy**2
        else:
            A[r,r] = 1.0

A.assemble()
print_matrix(A, "A MATRIX (u* and v*)")

# ==========================================================
# STEP 1 RHS
# ==========================================================
b_u = PETSc.Vec().createSeq(N)
b_v = PETSc.Vec().createSeq(N)

apply_bc(u,v)

for i in range(1,Ny-1):
    for j in range(1,Nx-1):

        dudx = (u[i,j] - u[i,j-1])/dx
        dudy = (u[i,j] - u[i-1,j])/dy

        dvdx = (v[i,j] - v[i,j-1])/dx
        dvdy = (v[i,j] - v[i-1,j])/dy

        b_u.setValue(idx(i,j), u[i,j] - dt*(u[i,j]*dudx + v[i,j]*dudy))
        b_v.setValue(idx(i,j), v[i,j] - dt*(u[i,j]*dvdx + v[i,j]*dvdy))

b_u.assemble()
b_v.assemble()

print("\n===== RHS u* =====\n", b_u.getArray())
print("\n===== RHS v* =====\n", b_v.getArray())

# ==========================================================
# SOLVE u*, v*
# ==========================================================
ksp = PETSc.KSP().create() 
ksp.setOperators(A)
ksp.setType('gmres')
ksp.getPC().setType('ilu')

x_u = PETSc.Vec().createSeq(N)
x_v = PETSc.Vec().createSeq(N)

ksp.solve(b_u, x_u)
ksp.solve(b_v, x_v)

u_star = x_u.getArray().reshape((Ny,Nx))
v_star = x_v.getArray().reshape((Ny,Nx))

# penalized body
u_star = u_star / (1 + dt*sigma)
v_star = v_star / (1 + dt*sigma)

print("\n===== u* =====\n", u_star)
print("\n===== v* =====\n", v_star)

# ==========================================================
# STEP 2: PRESSURE MATRIX (FIXED)
# ==========================================================
A_p = PETSc.Mat().createAIJ([N,N])

for i in range(Ny):
    for j in range(Nx):
        r = idx(i,j)

        if 1 <= i < Ny-1 and 1 <= j < Nx-1:
            A_p[r,r] = -2/dx**2 - 2/dy**2
            A_p[r, idx(i,j+1)] = 1/dx**2
            A_p[r, idx(i,j-1)] = 1/dx**2
            A_p[r, idx(i+1,j)] = 1/dy**2
            A_p[r, idx(i-1,j)] = 1/dy**2
        else:
            if j == Nx-1:
                # RIGHT → Dirichlet p=0
                A_p[r,r] = 1.0
            else:
                # Neumann elsewhere
                A_p[r,r] = 1.0
                if j < Nx-1:
                    A_p[r, idx(i,j+1)] = -1.0

A_p.assemble()
print_matrix(A_p, "PRESSURE MATRIX")

# ==========================================================
# STEP 2 RHS
# ==========================================================
b_p = PETSc.Vec().createSeq(N)

for i in range(1,Ny-1):
    for j in range(1,Nx-1):

        div = ((u_star[i,j+1]-u_star[i,j-1])/(2*dx) +
               (v_star[i+1,j]-v_star[i-1,j])/(2*dy))

        b_p.setValue(idx(i,j), div/dt)

b_p.assemble()

print("\n===== RHS PRESSURE =====\n", b_p.getArray())

# ==========================================================
# SOLVE PRESSURE
# ==========================================================
x_p = PETSc.Vec().createSeq(N)
ksp.setOperators(A_p)
ksp.solve(b_p, x_p)

p = x_p.getArray().reshape((Ny,Nx))

# enforce outlet pressure
p[:,-1] = 0.0

print("\n===== PRESSURE =====\n", p)

# ==========================================================
# STEP 3: VELOCITY CORRECTION
# ==========================================================
u_new = u_star.copy()
v_new = v_star.copy()

for i in range(1,Ny-1):
    for j in range(1,Nx-1):

        dpdx = (p[i,j+1]-p[i,j-1])/(2*dx)
        dpdy = (p[i+1,j]-p[i-1,j])/(2*dy)

        u_new[i,j] -= dt*dpdx
        v_new[i,j] -= dt*dpdy

# apply BC again
apply_bc(u_new, v_new)

print("\n===== FINAL u =====\n", u_new)
print("\n===== FINAL v =====\n", v_new)