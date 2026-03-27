#include "hit_module.hh"
#include "fluid_3d.hh"
#include "sim_params.hh"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdint.h>
#include <fftw3.h>

hit_diag_data::hit_diag_data():
    E_total(0), E_low(0), E_high(0), alpha(1), target_err(0),
    var_x(0), var_y(0), var_z(0), forcing_l2(0), div_before(0), div_after(0),
    eps(0), re_lambda(0), t_pack(0), t_fft_fwd(0), t_shell(0), t_fft_bwd(0), t_unpack(0) {}

hit_module::hit_module(fluid_3d &f3d_):
    f3d(f3d_), spars(f3d_.spars), comm(MPI_COMM_WORLD), rank(0), nprocs(1),
    hit_enable(false), initialized(false), maintain_after_insert(true),
    init_type(0), apply_stride(1), diag_stride(1), spectrum_stride(100),
    warmup_steps(0), insert_step(-1), insert_ramp_steps(0), seed_base(12345),
    project_modified_modes(true), project_all_modes(false),
    write_spectrum(true), write_energy(true),
    kf2(3.0), kd(50.0), target_re_lambda(84.0), target_energy(0.0),
    alpha_last(1.0), eta(0.0), nx(0), ny(0), nz(0), nloc(0),
    fh_energy(NULL), fh_divergence(NULL), fh_spectrum(NULL) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
}

hit_module::~hit_module() {
    if(fh_energy) fclose(fh_energy);
    if(fh_divergence) fclose(fh_divergence);
    if(fh_spectrum) fclose(fh_spectrum);
}

void hit_module::setup_hit_module() {
    hit_enable = spars->hit_enable;
    if(!hit_enable) return;
    maintain_after_insert = spars->hit_keep_forcing_after_insert;
    init_type = spars->hit_init_type;
    apply_stride = spars->hit_apply_stride;
    diag_stride = spars->hit_diag_stride;
    spectrum_stride = spars->hit_spectrum_stride;
    warmup_steps = spars->hit_warmup_steps;
    insert_step = spars->hit_insert_step;
    insert_ramp_steps = spars->hit_insert_ramp_steps;
    seed_base = spars->hit_seed;
    project_modified_modes = spars->hit_project_modified_modes;
    project_all_modes = spars->hit_project_all_modes;
    write_spectrum = spars->hit_write_spectrum;
    write_energy = spars->hit_write_energy;
    kf2 = spars->hit_kf2 * spars->hit_kf2;
    kd = spars->hit_kd;
    target_re_lambda = spars->hit_target_re_lambda;
    eta = (insert_step>=0)?0.0:1.0;
    nx = f3d.m; ny = f3d.n; nz = f3d.o;
    nloc = f3d.sm * f3d.sn * f3d.so;
    for(int c=0;c<3;c++) {
        ur[c].assign(nloc, 0.0);
        uh[c].assign(nloc, std::complex<double>(0.0, 0.0));
    }
    build_wavenumbers();
    if(rank==0 && write_energy) {
        char fn[512];
        sprintf(fn, "%s/hit_energy.dat", spars->dirname);
        fh_energy = p_safe_fopen(fn, "w");
        fprintf(fh_energy, "#t step E_total E_low E_high alpha target_err var_x var_y var_z forcing_l2 div_before div_after eps Re_lambda t_pack t_fft_fwd t_shell t_fft_bwd t_unpack\n");
        fflush(fh_energy);
        sprintf(fn, "%s/hit_divergence.dat", spars->dirname);
        fh_divergence = p_safe_fopen(fn, "w");
        fprintf(fh_divergence, "#t step div_before div_after alpha\n");
        fflush(fh_divergence);
    }
    if(rank==0 && write_spectrum) {
        char fn[512];
        sprintf(fn, "%s/hit_spectrum.dat", spars->dirname);
        fh_spectrum = p_safe_fopen(fn, "w");
        fprintf(fh_spectrum, "#step k Ek\n");
        fflush(fh_spectrum);
    }
}

double hit_module::mode_rand01(int gi, int gj, int gk, int c, int extra) const {
    uint64_t x = static_cast<uint64_t>(seed_base) + 0x9e3779b97f4a7c15ULL;
    x ^= static_cast<uint64_t>(gi + 4096) * 0xbf58476d1ce4e5b9ULL;
    x ^= static_cast<uint64_t>(gj + 8192) * 0x94d049bb133111ebULL;
    x ^= static_cast<uint64_t>(gk + 16384) * 0x2545F4914F6CDD1DULL;
    x ^= static_cast<uint64_t>(c + 1) * 0x632BE59BD9B4E019ULL;
    x ^= static_cast<uint64_t>(extra + 7) * 0xD6E8FEB86659FD93ULL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    const double inv = 1.0 / static_cast<double>(UINT64_MAX);
    return (x & 0xFFFFFFFFFFFFULL) * inv;
}

void hit_module::build_wavenumbers() {
    kx.assign(nloc, 0.0);
    ky.assign(nloc, 0.0);
    kz.assign(nloc, 0.0);
    k2.assign(nloc, 0.0);
    low_mask.assign(nloc, 0.0);
    int ind = 0;
    for(int kk=0; kk<f3d.so; ++kk){
        int gk = f3d.ak + kk;
        int kz_i = (gk <= nz/2) ? gk : (gk - nz);
        for(int jj=0; jj<f3d.sn; ++jj){
            int gj = f3d.aj + jj;
            int ky_i = (gj <= ny/2) ? gj : (gj - ny);
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                int gi = f3d.ai + ii;
                int kx_i = (gi <= nx/2) ? gi : (gi - nx);
                kx[ind] = static_cast<double>(kx_i);
                ky[ind] = static_cast<double>(ky_i);
                kz[ind] = static_cast<double>(kz_i);
                k2[ind] = kx[ind]*kx[ind] + ky[ind]*ky[ind] + kz[ind]*kz[ind];
                low_mask[ind] = (k2[ind] <= kf2) ? 1.0 : 0.0;
            }
        }
    }
}

void hit_module::local_fft_forward() {
    for(int c=0;c<3;c++){
        fftw_plan p = fftw_plan_dft_3d(f3d.sm, f3d.sn, f3d.so,
                                       reinterpret_cast<fftw_complex*>(&uh[c][0]),
                                       reinterpret_cast<fftw_complex*>(&uh[c][0]),
                                       FFTW_FORWARD, FFTW_ESTIMATE);
        for(int i=0;i<nloc;i++) uh[c][i] = std::complex<double>(ur[c][i], 0.0);
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
}

void hit_module::local_fft_backward() {
    const double norm = 1.0/static_cast<double>(nloc);
    for(int c=0;c<3;c++){
        fftw_plan p = fftw_plan_dft_3d(f3d.sm, f3d.sn, f3d.so,
                                       reinterpret_cast<fftw_complex*>(&uh[c][0]),
                                       reinterpret_cast<fftw_complex*>(&uh[c][0]),
                                       FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
        for(int i=0;i<nloc;i++) ur[c][i] = uh[c][i].real()*norm;
    }
}

void hit_module::project_divergence_free(bool modified_only) {
    for(int i=0;i<nloc;i++){
        if(k2[i] == 0.0) { uh[0][i]=uh[1][i]=uh[2][i]=0.0; continue; }
        if(modified_only && low_mask[i] < 0.5) continue;
        std::complex<double> dot = kx[i]*uh[0][i] + ky[i]*uh[1][i] + kz[i]*uh[2][i];
        dot /= k2[i];
        uh[0][i] -= dot*kx[i];
        uh[1][i] -= dot*ky[i];
        uh[2][i] -= dot*kz[i];
    }
}

double hit_module::kinetic_energy() {
    double local = 0.0;
    for(int i=0;i<nloc;i++){
        local += ur[0][i]*ur[0][i] + ur[1][i]*ur[1][i] + ur[2][i]*ur[2][i];
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return 0.5*global/static_cast<double>(nx*ny*nz);
}

double hit_module::l2_divergence() {
    const int sx = f3d.sm4;
    const int sy = f3d.sn4;
    const int sz = f3d.so4;
    double local = 0.0;
    for(int kk=2; kk<sz-2; ++kk){
        for(int jj=2; jj<sy-2; ++jj){
            for(int ii=2; ii<sx-2; ++ii){
                int id = f3d.index(ii,jj,kk);
                int ixp = f3d.index(ii+1,jj,kk), ixm = f3d.index(ii-1,jj,kk);
                int iyp = f3d.index(ii,jj+1,kk), iym = f3d.index(ii,jj-1,kk);
                int izp = f3d.index(ii,jj,kk+1), izm = f3d.index(ii,jj,kk-1);
                double div = (f3d.u_mem[ixp].vel[0]-f3d.u_mem[ixm].vel[0])/(2.0*f3d.dx)
                           + (f3d.u_mem[iyp].vel[1]-f3d.u_mem[iym].vel[1])/(2.0*f3d.dy)
                           + (f3d.u_mem[izp].vel[2]-f3d.u_mem[izm].vel[2])/(2.0*f3d.dz);
                local += div*div;
            }
        }
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sqrt(global/static_cast<double>(nx*ny*nz));
}

void hit_module::hit_initialize_random_phase() {
    if(!hit_enable) return;
    const double nu = spars->fmu;
    target_energy = target_re_lambda * (nu*kd)*(nu*kd)/sqrt(20.0/3.0);
    for(int i=0;i<nloc;i++){
        double km = sqrt(std::max(k2[i], 1.0));
        double ek_low = sqrt(9.0/11.0/sqrt(kf2) * k2[i]/std::max(kf2, 1e-12));
        double ek_hi = sqrt(9.0/11.0/sqrt(kf2) * pow(km/std::max(sqrt(kf2),1e-8), -5.0/3.0));
        double ek = (low_mask[i]>0.5)?ek_low:ek_hi;
        double kk = std::max(k2[i], 1.0);
        double amp = sqrt(std::max(ek, 0.0)/(4.0*M_PI*kk));
        for(int c=0;c<3;c++){
            double theta = 2.0*M_PI*mode_rand01(f3d.ai + (i%f3d.sm), f3d.aj + ((i/f3d.sm)%f3d.sn), f3d.ak + (i/(f3d.sm*f3d.sn)), c, 0);
            uh[c][i] = std::polar(amp, theta);
        }
    }
    project_divergence_free(false);
    local_fft_backward();
    double e0 = kinetic_energy();
    double fac = (e0>0.0)?sqrt(target_energy/e0):1.0;
    for(int i=0;i<nloc;i++) for(int c=0;c<3;c++) ur[c][i]*=fac;
    int ind = 0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                fp->vel[0] = ur[0][ind];
                fp->vel[1] = ur[1][ind];
                fp->vel[2] = ur[2][ind];
            }
        }
    }
    f3d.fill_boundary_cc(false);
    initialized = true;
}

void hit_module::hit_initialize_from_restart() {
    if(!hit_enable) return;
    FILE *fh = fopen(spars->hit_restart_file, "rb");
    if(!fh) p_fatal_error("hit_initialize_from_restart: cannot open restart file", 1);
    fread(&target_energy, sizeof(double), 1, fh);
    for(int c=0;c<3;c++) fread(&ur[c][0], sizeof(double), nloc, fh);
    fclose(fh);
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                fp->vel[0] = ur[0][ind];
                fp->vel[1] = ur[1][ind];
                fp->vel[2] = ur[2][ind];
            }
        }
    }
    f3d.fill_boundary_cc(false);
    initialized = true;
}

void hit_module::apply_hit_forcing_spectral_shell(int step) {
    if(!hit_enable || !initialized) return;
    if(step % apply_stride != 0) return;
    if(insert_step>=0 && step>=insert_step && !maintain_after_insert) return;
    double t0 = MPI_Wtime();
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                ur[0][ind] = fp->vel[0];
                ur[1][ind] = fp->vel[1];
                ur[2][ind] = fp->vel[2];
            }
        }
    }
    diag.t_pack = MPI_Wtime()-t0;
    diag.div_before = l2_divergence();
    t0 = MPI_Wtime();
    local_fft_forward();
    diag.t_fft_fwd = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    double El=0.0, Et=0.0;
    for(int i=0;i<nloc;i++){
        double mode_e = 0.5*(std::norm(uh[0][i]) + std::norm(uh[1][i]) + std::norm(uh[2][i]));
        Et += mode_e;
        El += low_mask[i]*mode_e;
    }
    MPI_Allreduce(MPI_IN_PLACE, &Et, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &El, 1, MPI_DOUBLE, MPI_SUM, comm);
    double Eh = Et - El;
    double alpha2 = (target_energy - Eh)/std::max(El, 1e-30);
    if(alpha2 < 0.0) alpha2 = 0.0;
    double alpha = sqrt(alpha2);
    alpha_last = alpha;
    for(int i=0;i<nloc;i++){
        double s = alpha*low_mask[i] + (1.0-low_mask[i]);
        for(int c=0;c<3;c++) uh[c][i] *= s;
    }
    if(project_all_modes) project_divergence_free(false);
    else if(project_modified_modes) project_divergence_free(true);
    diag.t_shell = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    local_fft_backward();
    diag.t_fft_bwd = MPI_Wtime()-t0;
    t0 = MPI_Wtime();
    ind=0;
    double vloc[3] = {0,0,0};
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                double du0 = ur[0][ind]-fp->vel[0];
                double du1 = ur[1][ind]-fp->vel[1];
                double du2 = ur[2][ind]-fp->vel[2];
                diag.forcing_l2 += du0*du0 + du1*du1 + du2*du2;
                fp->vel[0] = ur[0][ind];
                fp->vel[1] = ur[1][ind];
                fp->vel[2] = ur[2][ind];
                vloc[0] += ur[0][ind]*ur[0][ind];
                vloc[1] += ur[1][ind]*ur[1][ind];
                vloc[2] += ur[2][ind]*ur[2][ind];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, vloc, 3, MPI_DOUBLE, MPI_SUM, comm);
    f3d.fill_boundary_cc(false);
    diag.t_unpack = MPI_Wtime()-t0;
    diag.div_after = l2_divergence();
    diag.E_total = kinetic_energy();
    diag.E_low = El;
    diag.E_high = Eh;
    diag.alpha = alpha;
    diag.target_err = diag.E_total - target_energy;
    diag.var_x = vloc[0]/(nx*ny*nz);
    diag.var_y = vloc[1]/(nx*ny*nz);
    diag.var_z = vloc[2]/(nx*ny*nz);
    diag.forcing_l2 = sqrt(diag.forcing_l2/(nloc+1e-12));
    diag.eps = 2.0*spars->fmu*diag.E_total;
    diag.re_lambda = (diag.eps>1e-20)?sqrt(20.0*diag.E_total*diag.E_total/(3.0*spars->fmu*diag.eps)):0.0;
}

void hit_module::compute_shell_spectrum(std::vector<double> &ek) {
    int nb = static_cast<int>(sqrt((nx/2.0)*(nx/2.0)+(ny/2.0)*(ny/2.0)+(nz/2.0)*(nz/2.0))/sqrt(3.0))+1;
    ek.assign(nb, 0.0);
    std::vector<double> cnt(nb, 0.0);
    for(int i=0;i<nloc;i++){
        int kbin = static_cast<int>(sqrt(k2[i]) + 0.5);
        if(kbin>=0 && kbin<nb){
            ek[kbin] += 0.5*(std::norm(uh[0][i])+std::norm(uh[1][i])+std::norm(uh[2][i]));
            cnt[kbin] += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &ek[0], nb, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &cnt[0], nb, MPI_DOUBLE, MPI_SUM, comm);
    for(int k=0;k<nb;k++) if(cnt[k]>0.0) ek[k] /= cnt[k];
}

void hit_module::write_hit_diagnostics(int step) {
    if(!hit_enable || !initialized) return;
    if(!write_energy || step % diag_stride != 0) return;
    if(rank==0 && fh_energy){
        fprintf(fh_energy, "%.12e %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.6e %.6e %.6e %.6e %.6e\n",
                f3d.time, step, diag.E_total, diag.E_low, diag.E_high, diag.alpha, diag.target_err,
                diag.var_x, diag.var_y, diag.var_z, diag.forcing_l2, diag.div_before, diag.div_after,
                diag.eps, diag.re_lambda, diag.t_pack, diag.t_fft_fwd, diag.t_shell, diag.t_fft_bwd, diag.t_unpack);
        fflush(fh_energy);
    }
    if(rank==0 && fh_divergence){
        fprintf(fh_divergence, "%.12e %d %.12e %.12e %.12e\n", f3d.time, step, diag.div_before, diag.div_after, diag.alpha);
        fflush(fh_divergence);
    }
    if(write_spectrum && step % spectrum_stride == 0){
        std::vector<double> ek;
        compute_shell_spectrum(ek);
        if(rank==0 && fh_spectrum){
            for(size_t k=0;k<ek.size();++k) fprintf(fh_spectrum, "%d %zu %.12e\n", step, k, ek[k]);
            fprintf(fh_spectrum, "\n");
            fflush(fh_spectrum);
        }
    }
}

void hit_module::begin_particle_insertion(int step) {
    if(insert_step < 0 || step != insert_step) return;
    f3d.mgmt->particles_inserted = true;
    f3d.mgmt->set_insertion_eta(0.0);
    f3d.init_refmap();
    f3d.init_extrapolate(false);
    f3d.fill_boundary_cc(false);
    f3d.pp_solve(false);
}

void hit_module::update_particle_insertion_ramp(int step) {
    if(insert_step < 0 || step < insert_step) return;
    if(insert_ramp_steps <= 0){
        eta = 1.0;
    } else {
        double s = std::min(1.0, std::max(0.0, (step - insert_step + 1.0)/insert_ramp_steps));
        eta = s*s*(3.0-2.0*s);
    }
    f3d.mgmt->set_insertion_eta(eta);
}

void hit_module::write_restart_state(FILE *fh) {
    if(!hit_enable) return;
    fwrite(&target_energy, sizeof(double), 1, fh);
    fwrite(&alpha_last, sizeof(double), 1, fh);
    fwrite(&eta, sizeof(double), 1, fh);
}

void hit_module::read_restart_state(FILE *fh) {
    if(!hit_enable) return;
    fread(&target_energy, sizeof(double), 1, fh);
    fread(&alpha_last, sizeof(double), 1, fh);
    fread(&eta, sizeof(double), 1, fh);
    f3d.mgmt->set_insertion_eta(eta);
}

