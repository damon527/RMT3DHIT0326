#ifndef HIT_MODULE_HH
#define HIT_MODULE_HH

#include <vector>
#include <complex>
#include <cstdio>
#include <mpi.h>

class fluid_3d;
class sim_params;

struct hit_diag_data {
    double E_total, E_low, E_high;
    double alpha, target_err;
    double var_x, var_y, var_z;
    double forcing_l2;
    double div_before, div_after;
    double eps, re_lambda;
    double t_pack, t_fft_fwd, t_shell, t_fft_bwd, t_unpack;
    hit_diag_data();
};

class hit_module {
public:
    explicit hit_module(fluid_3d &f3d_);
    ~hit_module();

    bool enabled() const { return hit_enable; }
    void setup_hit_module();
    void hit_initialize_random_phase();
    void hit_initialize_from_restart();
    void apply_hit_forcing_spectral_shell(int step);
    void write_hit_diagnostics(int step);
    void begin_particle_insertion(int step);
    void update_particle_insertion_ramp(int step);
    void write_restart_state(FILE *fh);
    void read_restart_state(FILE *fh);

private:
    fluid_3d &f3d;
    sim_params *spars;
    MPI_Comm comm;
    int rank, nprocs;
    bool hit_enable;
    bool initialized;
    bool maintain_after_insert;
    int init_type;
    int apply_stride, diag_stride, spectrum_stride;
    int warmup_steps, insert_step, insert_ramp_steps;
    int seed_base;
    bool project_modified_modes, project_all_modes;
    bool write_spectrum, write_energy;
    double kf2, kd, target_re_lambda, target_energy;
    double alpha_last, eta;
    int nx, ny, nz, nloc;
    std::vector<std::complex<double> > uh[3];
    std::vector<double> ur[3];
    std::vector<double> kx, ky, kz, k2, low_mask;
    hit_diag_data diag;
    FILE *fh_energy;
    FILE *fh_divergence;
    FILE *fh_spectrum;

    void build_wavenumbers();
    void project_divergence_free(bool modified_only);
    void local_fft_forward();
    void local_fft_backward();
    void compute_shell_spectrum(std::vector<double> &ek);
    double mode_rand01(int gi, int gj, int gk, int c, int extra) const;
    double l2_divergence();
    double kinetic_energy();
};

#endif
