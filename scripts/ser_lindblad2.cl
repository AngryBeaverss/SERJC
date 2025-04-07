__kernel void ser_lindblad(
    __global double2 *rho,
    __global double2 *H,
    __global double2 *L,
    __global double2 *I,
    double hbar,
    double gamma,
    double beta,
    double dt)
{
    int gid = get_global_id(0);
    if (gid >= 16) return;  // Ensure gid is within bounds for a 4x4 system

    // Map gid to matrix indices
    int i = gid / 4;
    int j = gid % 4;

    // Load matrices into private memory
    double2 rho_private[16];
    double2 H_private[16];
    double2 L_private[16];
    double2 I_private[16];
    for (int k = 0; k < 16; k++) {
        rho_private[k] = rho[k];
        H_private[k]   = H[k];
        L_private[k]   = L[k];
        I_private[k]   = I[k];
    }

    // Compute commutator [H, rho]
    double2 H_rho_ij = (double2)(0.0, 0.0);
    double2 rho_H_ij = (double2)(0.0, 0.0);
    for (int k = 0; k < 4; k++) {
        int idx_ik = i * 4 + k;
        int idx_kj = k * 4 + j;
        H_rho_ij += H_private[idx_ik] * rho_private[idx_kj];
        rho_H_ij += rho_private[idx_ik] * H_private[idx_kj];
    }
    double2 comm_real = (H_rho_ij - rho_H_ij) / hbar;
    double2 commutator = (double2)(-comm_real.y, comm_real.x); // Equivalent to multiplying by i

    // Compute Lindblad dissipator with correct conjugation
    double2 L_rho_Ldag = (double2)(0.0, 0.0);
    double2 Ldag_L_rho = (double2)(0.0, 0.0);
    double2 rho_Ldag_L = (double2)(0.0, 0.0);
    for (int k = 0; k < 4; k++) {
        int idx_ik = i * 4 + k;
        int idx_kj = k * 4 + j;
        double2 L_conj_kj = (double2)(L_private[idx_kj].x, -L_private[idx_kj].y);
        double2 L_conj_ik = (double2)(L_private[idx_ik].x, -L_private[idx_ik].y);
        L_rho_Ldag += L_private[idx_ik] * rho_private[idx_kj] * L_conj_kj;
        Ldag_L_rho += L_conj_ik * L_private[idx_kj] * rho_private[idx_kj];
        rho_Ldag_L += rho_private[idx_ik] * L_conj_kj * L_private[idx_kj];
    }
    double2 lindblad_term = gamma * (L_rho_Ldag - 0.5 * (Ldag_L_rho + rho_Ldag_L));

    // Compute feedback function F(rho) = exp(-coherence_squared)
    double coherence_squared = 0.0;
    for (int ii = 0; ii < 4; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            if (ii != jj) {
                int idx = ii * 4 + jj;
                coherence_squared += rho_private[idx].x * rho_private[idx].x +
                                     rho_private[idx].y * rho_private[idx].y;
            }
        }
    }
    double F_rho = exp(-coherence_squared);

    // Compute (I - rho)
    double2 I_minus_rho[16];
    for (int k = 0; k < 16; k++) {
        I_minus_rho[k] = I_private[k] - rho_private[k];
    }

    // Compute L * rho
    double2 L_rho[16];
    for (int k = 0; k < 16; k++) {
        L_rho[k] = (double2)(0.0, 0.0);
        for (int mm = 0; mm < 4; mm++) {
            int idx_km = (k / 4) * 4 + mm;
            int idx_mj = mm * 4 + (k % 4);
            L_rho[k] += L_private[idx_km] * rho_private[idx_mj];
        }
    }

    // Compute L * rho * Ldag
    double2 L_rho_Ldag_new[16];
    for (int k = 0; k < 16; k++) {
        L_rho_Ldag_new[k] = (double2)(0.0, 0.0);
        for (int mm = 0; mm < 4; mm++) {
            int idx_km = (k / 4) * 4 + mm;
            int idx_mj = mm * 4 + (k % 4);
            L_rho_Ldag_new[k] += L_rho[idx_km] * L_private[idx_mj];
        }
    }

    // Compute SER feedback term: (I - rho) * L_rho_Ldag_new * (I - rho)
    double2 temp[16] = { (double2)(0.0, 0.0) };
    for (int ii = 0; ii < 4; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            int idx_ij = ii * 4 + jj;
            for (int mm = 0; mm < 4; mm++) {
                int idx_im = ii * 4 + mm;
                int idx_mj = mm * 4 + jj;
                temp[idx_ij] += I_minus_rho[idx_im] * L_rho_Ldag_new[idx_mj];
                            }
        }
    }

    double2 SER_feedback[16] = { (double2)(0.0, 0.0) };
    for (int ii = 0; ii < 4; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            int idx_ij = ii * 4 + jj;
            for (int nn = 0; nn < 4; nn++) {
                int idx_in = ii * 4 + nn;
                int idx_nj = nn * 4 + jj;
                SER_feedback[idx_ij] += temp[idx_in] * I_minus_rho[idx_nj];
            }
            SER_feedback[idx_ij] *= beta * F_rho;
        }
    }

    // Compute total derivative of rho
    double2 drho_dt = (-1.0 / hbar) * commutator + lindblad_term + SER_feedback[gid];

    // Update density matrix
    barrier(CLK_GLOBAL_MEM_FENCE);
    rho[gid] += dt * drho_dt;

// Enforce Hermitian symmetry
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        int idx = i * 4 + j;
        int idx_T = j * 4 + i;
        rho[idx] = (rho[idx] + (double2)(rho[idx_T].x, -rho[idx_T].y)) / 2.0;
    }
}

// Ensure rho remains positive semi-definite
for (int k = 0; k < 16; k++) {
    if (rho[k].x < 0) {
        rho[k] = (double2)(0.0, rho[k].y);
    }
}
// Compute the trace of rho
double trace_rho = 0.0;
for (int k = 0; k < 16; k++) {
    trace_rho += rho[k].x;  // Only sum the real parts
}

// Normalize rho to enforce Tr(rho) = 1
if (trace_rho != 0.0) {
    for (int k = 0; k < 16; k++) {
        rho[k] = rho[k] / trace_rho;  // Scale elements
    }
}
    barrier(CLK_GLOBAL_MEM_FENCE);
}
