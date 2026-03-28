#include "hit_module.hh"
#include "fluid_3d.hh"
#include "sim_params.hh"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdint.h>

hit_diag_data::hit_diag_data():
    E_total(0), E_low(0), E_high(0), alpha(1), target_err(0),
    var_x(0), var_y(0), var_z(0), forcing_l2(0), div_before(0), div_after(0),
    eps(0), re_lambda(0), t_pack(0), t_comm_g2s(0), t_fft_fwd(0), t_shell(0), t_fft_bwd(0), t_comm_s2g(0), t_unpack(0) {}

void hit_diag_data::reset_step(){
    forcing_l2 = 0.0;
    t_pack = t_comm_g2s = t_fft_fwd = t_shell = t_fft_bwd = t_comm_s2g = t_unpack = 0.0;
}

hit_module::hit_module(fluid_3d &f3d_):
    f3d(f3d_), spars(f3d_.spars), comm(MPI_COMM_WORLD), rank(0), nprocs(1),
    hit_enable(false), initialized(false), maintain_after_insert(true),
    init_type(0), apply_stride(1), cfl_recheck_stride(1), diag_stride(1), spectrum_stride(100),
    warmup_steps(0), insert_step(-1), insert_ramp_steps(0), forcing_calls(0), seed_base(12345),
    project_modified_modes(true), project_all_modes(false),
    write_spectrum(true), write_energy(true),
    kf2(3.0), kd(50.0), target_re_lambda(84.0), target_energy(0.0), target_energy_dyn(0.0),
    alpha_last(1.0), eta(0.0),
    nx(0), ny(0), nz(0), local_n0(0), local_0_start(0), alloc_local(0),
    fh_energy(NULL), fh_divergence(NULL), fh_spectrum(NULL) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    for(int c=0;c<3;c++){
        uh_raw[c] = NULL;
        plan_fwd[c] = NULL;
        plan_bwd[c] = NULL;
    }
}

hit_module::~hit_module() {
    destroy_fft_plans();
    if(fh_energy) fclose(fh_energy);
    if(fh_divergence) fclose(fh_divergence);
    if(fh_spectrum) fclose(fh_spectrum);
}

void hit_module::destroy_fft_plans(){
    for(int c=0;c<3;c++){
        if(plan_fwd[c]) fftw_destroy_plan(plan_fwd[c]);
        if(plan_bwd[c]) fftw_destroy_plan(plan_bwd[c]);
        plan_fwd[c] = plan_bwd[c] = NULL;
        if(uh_raw[c]) fftw_free(uh_raw[c]);
        uh_raw[c] = NULL;
    }
}

void hit_module::setup_fft_plans(){
    fftw_mpi_init();
    alloc_local = fftw_mpi_local_size_3d(nx, ny, nz, comm, &local_n0, &local_0_start);
    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    for(int c=0;c<3;c++){
        uh_raw[c] = fftw_alloc_complex(alloc_local);
        for(ptrdiff_t i=0;i<alloc_local;i++) uh_raw[c][i][0] = uh_raw[c][i][1] = 0.0;
        plan_fwd[c] = fftw_mpi_plan_dft_3d(nx, ny, nz, uh_raw[c], uh_raw[c], comm, FFTW_FORWARD, FFTW_MEASURE);
        plan_bwd[c] = fftw_mpi_plan_dft_3d(nx, ny, nz, uh_raw[c], uh_raw[c], comm, FFTW_BACKWARD, FFTW_MEASURE);
        uh[c].assign(nloc, std::complex<double>(0.0, 0.0));
    }
}

void hit_module::setup_hit_module() {
    hit_enable = spars->hit_enable;
    if(!hit_enable) return;

    maintain_after_insert = spars->hit_keep_forcing_after_insert;
    init_type = spars->hit_init_type;
    apply_stride = spars->hit_apply_stride;
    cfl_recheck_stride = spars->hit_cfl_recheck_stride;
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

    nx = f3d.m;
    ny = f3d.n;
    nz = f3d.o;

    int nloc_block = f3d.sm * f3d.sn * f3d.so;
    for(int c=0;c<3;c++) ur_block[c].assign(nloc_block, 0.0);

    setup_fft_plans();

    // gather slab x ownership
    std::vector<int> starts(nprocs), lens(nprocs);
    int st = static_cast<int>(local_0_start), ln = static_cast<int>(local_n0);
    MPI_Allgather(&st, 1, MPI_INT, &starts[0], 1, MPI_INT, comm);
    MPI_Allgather(&ln, 1, MPI_INT, &lens[0], 1, MPI_INT, comm);
    slab_x_start.resize(nprocs);
    slab_x_end.resize(nprocs);
    for(int r=0;r<nprocs;r++){
        slab_x_start[r] = starts[r];
        slab_x_end[r] = starts[r] + lens[r];
    }

    // gather block ownership for unpack
    int my_blk[6] = {f3d.ai, f3d.aj, f3d.ak, f3d.bi, f3d.bj, f3d.bk};
    std::vector<int> all_blk(6*nprocs, 0);
    MPI_Allgather(my_blk, 6, MPI_INT, &all_blk[0], 6, MPI_INT, comm);
    blk_ai.resize(nprocs); blk_aj.resize(nprocs); blk_ak.resize(nprocs);
    blk_bi.resize(nprocs); blk_bj.resize(nprocs); blk_bk.resize(nprocs);
    for(int r=0;r<nprocs;r++){
        blk_ai[r] = all_blk[6*r+0];
        blk_aj[r] = all_blk[6*r+1];
        blk_ak[r] = all_blk[6*r+2];
        blk_bi[r] = all_blk[6*r+3];
        blk_bj[r] = all_blk[6*r+4];
        blk_bk[r] = all_blk[6*r+5];
    }

    build_wavenumbers();

    if(rank==0 && write_energy) {
        char fn[512];
        sprintf(fn, "%s/hit_energy.dat", spars->dirname);
        fh_energy = p_safe_fopen(fn, "w");
        fprintf(fh_energy, "#t step E_total E_low E_high alpha target_err var_x var_y var_z forcing_l2 div_before div_after eps Re_lambda t_pack t_comm_g2s t_fft_fwd t_shell t_fft_bwd t_comm_s2g t_unpack\n");
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
    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    kx.assign(nloc, 0.0);
    ky.assign(nloc, 0.0);
    kz.assign(nloc, 0.0);
    k2.assign(nloc, 0.0);
    low_mask.assign(nloc, 0.0);

    ptrdiff_t id=0;
    for(ptrdiff_t ii=0; ii<local_n0; ++ii){
        int gi = static_cast<int>(local_0_start + ii);
        int kx_i = (gi <= nx/2) ? gi : (gi - nx);
        for(int jj=0; jj<ny; ++jj){
            int ky_i = (jj <= ny/2) ? jj : (jj - ny);
            for(int kk=0; kk<nz; ++kk, ++id){
                int kz_i = (kk <= nz/2) ? kk : (kk - nz);
                kx[id] = static_cast<double>(kx_i);
                ky[id] = static_cast<double>(ky_i);
                kz[id] = static_cast<double>(kz_i);
                k2[id] = kx[id]*kx[id] + ky[id]*ky[id] + kz[id]*kz[id];
                low_mask[id] = (k2[id] <= kf2) ? 1.0 : 0.0;
            }
        }
    }
}

void hit_module::mask_zero_nyquist(){
    ptrdiff_t id=0;
    for(ptrdiff_t ii=0; ii<local_n0; ++ii){
        int gi = static_cast<int>(local_0_start + ii);
        for(int jj=0; jj<ny; ++jj){
            for(int kk=0; kk<nz; ++kk, ++id){
                bool is_zero = (gi==0 && jj==0 && kk==0);
                bool nx_nyq = (nx%2==0 && gi==nx/2);
                bool ny_nyq = (ny%2==0 && jj==ny/2);
                bool nz_nyq = (nz%2==0 && kk==nz/2);
                if(is_zero || nx_nyq || ny_nyq || nz_nyq){
                    uh[0][id] = uh[1][id] = uh[2][id] = std::complex<double>(0.0, 0.0);
                }
            }
        }
    }
}

void hit_module::project_divergence_free(bool modified_only) {
    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    for(ptrdiff_t i=0;i<nloc;i++){
        if(k2[i] <= 0.0) { uh[0][i]=uh[1][i]=uh[2][i]=0.0; continue; }
        if(modified_only && low_mask[i] < 0.5) continue;
        std::complex<double> dot = kx[i]*uh[0][i] + ky[i]*uh[1][i] + kz[i]*uh[2][i];
        dot /= k2[i];
        uh[0][i] -= dot*kx[i];
        uh[1][i] -= dot*ky[i];
        uh[2][i] -= dot*kz[i];
    }
}

void hit_module::pack_block_velocity(){
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                ur_block[0][ind] = fp->vel[0];
                ur_block[1][ind] = fp->vel[1];
                ur_block[2][ind] = fp->vel[2];
            }
        }
    }
}

void hit_module::unpack_block_velocity(){
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                fp->vel[0] = ur_block[0][ind];
                fp->vel[1] = ur_block[1][ind];
                fp->vel[2] = ur_block[2][ind];
            }
        }
    }
}

int hit_module::slab_owner_from_x(int gi) const {
    for(int r=0;r<nprocs;r++) if(gi>=slab_x_start[r] && gi<slab_x_end[r]) return r;
    return 0;
}

int hit_module::block_owner_from_ijk(int gi, int gj, int gk) const {
    for(int r=0;r<nprocs;r++){
        if(gi>=blk_ai[r] && gi<blk_bi[r] && gj>=blk_aj[r] && gj<blk_bj[r] && gk>=blk_ak[r] && gk<blk_bk[r]) return r;
    }
    return 0;
}

void hit_module::redistribute_block_to_slab(){
    std::vector< std::vector<double> > sendv(nprocs);
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        int gk = f3d.ak + kk;
        for(int jj=0; jj<f3d.sn; ++jj){
            int gj = f3d.aj + jj;
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                int gi = f3d.ai + ii;
                int dst = slab_owner_from_x(gi);
                sendv[dst].push_back(static_cast<double>(gi));
                sendv[dst].push_back(static_cast<double>(gj));
                sendv[dst].push_back(static_cast<double>(gk));
                sendv[dst].push_back(ur_block[0][ind]);
                sendv[dst].push_back(ur_block[1][ind]);
                sendv[dst].push_back(ur_block[2][ind]);
            }
        }
    }

    std::vector<int> scnt(nprocs,0), rcnt(nprocs,0), sdisp(nprocs,0), rdisp(nprocs,0);
    for(int r=0;r<nprocs;r++) scnt[r] = static_cast<int>(sendv[r].size());
    MPI_Alltoall(&scnt[0],1,MPI_INT,&rcnt[0],1,MPI_INT,comm);
    for(int r=1;r<nprocs;r++) { sdisp[r]=sdisp[r-1]+scnt[r-1]; rdisp[r]=rdisp[r-1]+rcnt[r-1]; }
    std::vector<double> sbuf(sdisp[nprocs-1]+scnt[nprocs-1],0.0), rbuf(rdisp[nprocs-1]+rcnt[nprocs-1],0.0);
    for(int r=0;r<nprocs;r++) std::copy(sendv[r].begin(), sendv[r].end(), sbuf.begin()+sdisp[r]);

    MPI_Alltoallv(&sbuf[0], &scnt[0], &sdisp[0], MPI_DOUBLE,
                  &rbuf[0], &rcnt[0], &rdisp[0], MPI_DOUBLE, comm);

    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    for(ptrdiff_t i=0;i<nloc;i++) uh[0][i]=uh[1][i]=uh[2][i]=std::complex<double>(0.0,0.0);

    for(size_t p=0; p+5<rbuf.size(); p+=6){
        int gi = static_cast<int>(rbuf[p+0]+0.5);
        int gj = static_cast<int>(rbuf[p+1]+0.5);
        int gk = static_cast<int>(rbuf[p+2]+0.5);
        ptrdiff_t il = gi - static_cast<int>(local_0_start);
        if(il<0 || il>=local_n0) continue;
        ptrdiff_t id = (il*ny + gj)*nz + gk;
        uh[0][id] = std::complex<double>(rbuf[p+3], 0.0);
        uh[1][id] = std::complex<double>(rbuf[p+4], 0.0);
        uh[2][id] = std::complex<double>(rbuf[p+5], 0.0);
    }
}

void hit_module::redistribute_slab_to_block(){
    std::vector< std::vector<double> > sendv(nprocs);
    ptrdiff_t id=0;
    for(ptrdiff_t ii=0; ii<local_n0; ++ii){
        int gi = static_cast<int>(local_0_start + ii);
        for(int jj=0; jj<ny; ++jj){
            for(int kk=0; kk<nz; ++kk, ++id){
                int dst = block_owner_from_ijk(gi, jj, kk);
                sendv[dst].push_back(static_cast<double>(gi));
                sendv[dst].push_back(static_cast<double>(jj));
                sendv[dst].push_back(static_cast<double>(kk));
                sendv[dst].push_back(uh[0][id].real());
                sendv[dst].push_back(uh[1][id].real());
                sendv[dst].push_back(uh[2][id].real());
            }
        }
    }

    std::vector<int> scnt(nprocs,0), rcnt(nprocs,0), sdisp(nprocs,0), rdisp(nprocs,0);
    for(int r=0;r<nprocs;r++) scnt[r] = static_cast<int>(sendv[r].size());
    MPI_Alltoall(&scnt[0],1,MPI_INT,&rcnt[0],1,MPI_INT,comm);
    for(int r=1;r<nprocs;r++) { sdisp[r]=sdisp[r-1]+scnt[r-1]; rdisp[r]=rdisp[r-1]+rcnt[r-1]; }
    std::vector<double> sbuf(sdisp[nprocs-1]+scnt[nprocs-1],0.0), rbuf(rdisp[nprocs-1]+rcnt[nprocs-1],0.0);
    for(int r=0;r<nprocs;r++) std::copy(sendv[r].begin(), sendv[r].end(), sbuf.begin()+sdisp[r]);

    MPI_Alltoallv(&sbuf[0], &scnt[0], &sdisp[0], MPI_DOUBLE,
                  &rbuf[0], &rcnt[0], &rdisp[0], MPI_DOUBLE, comm);

    for(size_t p=0; p+5<rbuf.size(); p+=6){
        int gi = static_cast<int>(rbuf[p+0]+0.5);
        int gj = static_cast<int>(rbuf[p+1]+0.5);
        int gk = static_cast<int>(rbuf[p+2]+0.5);
        int ii = gi - f3d.ai;
        int jj = gj - f3d.aj;
        int kk = gk - f3d.ak;
        if(ii<0 || ii>=f3d.sm || jj<0 || jj>=f3d.sn || kk<0 || kk>=f3d.so) continue;
        int ind = ii + f3d.sm*(jj + f3d.sn*kk);
        ur_block[0][ind] = rbuf[p+3];
        ur_block[1][ind] = rbuf[p+4];
        ur_block[2][ind] = rbuf[p+5];
    }
}

double hit_module::kinetic_energy_from_slab() {
    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    double local = 0.0;
    for(ptrdiff_t i=0;i<nloc;i++) local += std::norm(uh[0][i]) + std::norm(uh[1][i]) + std::norm(uh[2][i]);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return 0.5*global/static_cast<double>(nx*ny*nz*nx*ny*nz);
}

double hit_module::l2_divergence() {
    const int sx = f3d.sm4, sy = f3d.sn4, sz = f3d.so4;
    double local = 0.0;
    for(int kk=2; kk<sz-2; ++kk){
        for(int jj=2; jj<sy-2; ++jj){
            for(int ii=2; ii<sx-2; ++ii){
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

void hit_module::update_dt_from_cfl(){
    double vloc = 0.0;
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                double u = ur_block[0][ind], v = ur_block[1][ind], w = ur_block[2][ind];
                double vm = sqrt(u*u+v*v+w*w);
                if(vm > vloc) vloc = vm;
            }
        }
    }
    double vmax=0.0;
    MPI_Allreduce(&vloc, &vmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    f3d.mgmt->vmax = vmax;
    f3d.mgmt->obligatory_cfl_recheck();
    f3d.set_dt(f3d.mgmt->dt_reg);
}

void hit_module::hit_initialize_random_phase() {
    if(!hit_enable) return;
    const double nu = spars->fmu;
    target_energy = target_re_lambda * (nu*kd)*(nu*kd)/sqrt(20.0/3.0);
    target_energy_dyn = target_energy;

    ptrdiff_t id=0;
    for(ptrdiff_t ii=0; ii<local_n0; ++ii){
        int gi = static_cast<int>(local_0_start + ii);
        for(int jj=0; jj<ny; ++jj){
            for(int kk=0; kk<nz; ++kk, ++id){
                double km = sqrt(std::max(k2[id], 1.0));
                double ek_low = sqrt(9.0/11.0/sqrt(kf2) * k2[id]/std::max(kf2, 1e-12));
                double ek_hi = sqrt(9.0/11.0/sqrt(kf2) * pow(km/std::max(sqrt(kf2),1e-8), -5.0/3.0));
                double ek = (low_mask[id]>0.5)?ek_low:ek_hi;
                double kk2 = std::max(k2[id], 1.0);
                double amp = sqrt(std::max(ek, 0.0)/(4.0*M_PI*kk2));
                for(int c=0;c<3;c++){
                    double theta = 2.0*M_PI*mode_rand01(gi, jj, kk, c, 0);
                    uh[c][id] = std::polar(amp, theta);
                }
            }
        }
    }
    project_divergence_free(false);
    mask_zero_nyquist();

    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    for(int c=0;c<3;c++){
        for(ptrdiff_t i=0;i<nloc;i++) { uh_raw[c][i][0]=uh[c][i].real(); uh_raw[c][i][1]=uh[c][i].imag(); }
        fftw_execute(plan_bwd[c]);
        for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] = std::complex<double>(uh_raw[c][i][0], uh_raw[c][i][1]);
    }
    const double norm = 1.0/static_cast<double>(nx*ny*nz);
    for(int c=0;c<3;c++) for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] *= norm;

    // scale energy to target
    double e0 = 0.0;
    for(ptrdiff_t i=0;i<nloc;i++) e0 += std::norm(uh[0][i]) + std::norm(uh[1][i]) + std::norm(uh[2][i]);
    MPI_Allreduce(MPI_IN_PLACE, &e0, 1, MPI_DOUBLE, MPI_SUM, comm);
    e0 *= 0.5/static_cast<double>(nx*ny*nz);
    double fac = (e0>0.0)?sqrt(target_energy/e0):1.0;
    for(int c=0;c<3;c++) for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] *= fac;

    redistribute_slab_to_block();
    unpack_block_velocity();
    f3d.fill_boundary_cc(false);
    update_dt_from_cfl();
    initialized = true;
}

void hit_module::hit_initialize_from_restart() {
    if(!hit_enable) return;
    FILE *fh = fopen(spars->hit_restart_file, "rb");
    if(!fh) p_fatal_error("hit_initialize_from_restart: cannot open restart file", 1);
    size_t ok = fread(&target_energy, sizeof(double), 1, fh);
    if(ok!=1) p_fatal_error("hit_initialize_from_restart: failed reading target_energy", 1);
    target_energy_dyn = target_energy;
    for(int c=0;c<3;c++){
        ok = fread(&ur_block[c][0], sizeof(double), ur_block[c].size(), fh);
        if(ok!=ur_block[c].size()) p_fatal_error("hit_initialize_from_restart: failed reading velocity buffer", 1);
    }
    fclose(fh);

    unpack_block_velocity();
    f3d.fill_boundary_cc(false);
    update_dt_from_cfl();
    initialized = true;
}

void hit_module::apply_hit_forcing_spectral_shell(int step) {
    if(!hit_enable || !initialized) return;
    if(step % apply_stride != 0) return;
    if(insert_step>=0 && step>=insert_step && !maintain_after_insert) return;
    forcing_calls++;

    const bool do_diag_sample = (write_energy && (step % diag_stride == 0));

    diag.reset_step();

    double t0 = MPI_Wtime();
    pack_block_velocity();
    diag.t_pack = MPI_Wtime()-t0;

    if(do_diag_sample) diag.div_before = l2_divergence();
    else diag.div_before = 0.0;

    t0 = MPI_Wtime();
    redistribute_block_to_slab();
    diag.t_comm_g2s = MPI_Wtime()-t0;

    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    t0 = MPI_Wtime();
    for(int c=0;c<3;c++){
        for(ptrdiff_t i=0;i<nloc;i++) { uh_raw[c][i][0]=uh[c][i].real(); uh_raw[c][i][1]=uh[c][i].imag(); }
        fftw_execute(plan_fwd[c]);
        for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] = std::complex<double>(uh_raw[c][i][0], uh_raw[c][i][1]);
    }
    diag.t_fft_fwd = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    double El=0.0, Et=0.0;
    for(ptrdiff_t i=0;i<nloc;i++){
        double mode_e = 0.5*(std::norm(uh[0][i]) + std::norm(uh[1][i]) + std::norm(uh[2][i]));
        Et += mode_e;
        El += low_mask[i]*mode_e;
    }
    MPI_Allreduce(MPI_IN_PLACE, &Et, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &El, 1, MPI_DOUBLE, MPI_SUM, comm);
    double Eh = Et - El;

    // gradual target approach + safety clamps
    const double relax = 0.05;
    target_energy_dyn += relax*(target_energy - target_energy_dyn);
    const double El_floor = std::max(1e-12*target_energy, 1e-16);
    double alpha2 = (target_energy_dyn - Eh)/std::max(El, El_floor);
    if(alpha2 < 0.0) alpha2 = 0.0;
    double alpha = sqrt(alpha2);
    const double alpha_max = 1.10;
    if(alpha > alpha_max) alpha = alpha_max;
    alpha_last = alpha;

    for(ptrdiff_t i=0;i<nloc;i++){
        double s = alpha*low_mask[i] + (1.0-low_mask[i]);
        uh[0][i] *= s;
        uh[1][i] *= s;
        uh[2][i] *= s;
    }

    if(project_all_modes) project_divergence_free(false);
    else if(project_modified_modes) project_divergence_free(true);
    mask_zero_nyquist();

    double diss_local = 0.0;
    for(ptrdiff_t i=0;i<nloc;i++){
        double v2_hat = std::norm(uh[0][i]) + std::norm(uh[1][i]) + std::norm(uh[2][i]);
        diss_local += k2[i]*v2_hat;
    }
    MPI_Allreduce(MPI_IN_PLACE, &diss_local, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double nxyz = static_cast<double>(nx)*static_cast<double>(ny)*static_cast<double>(nz);
    const double eps_spec = spars->fmu * diss_local/(nxyz*nxyz);

    diag.t_shell = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    for(int c=0;c<3;c++){
        fftw_execute(plan_bwd[c]);
        for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] = std::complex<double>(uh_raw[c][i][0], uh_raw[c][i][1]);
    }
    const double norm = 1.0/static_cast<double>(nx*ny*nz);
    for(int c=0;c<3;c++) for(ptrdiff_t i=0;i<nloc;i++) uh[c][i] *= norm;
    diag.t_fft_bwd = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    redistribute_slab_to_block();
    diag.t_comm_s2g = MPI_Wtime()-t0;

    t0 = MPI_Wtime();
    double vloc[3] = {0,0,0};
    int ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                field *fp = f3d.u0 + f3d.index(ii,jj,kk);
                double du0 = ur_block[0][ind]-fp->vel[0];
                double du1 = ur_block[1][ind]-fp->vel[1];
                double du2 = ur_block[2][ind]-fp->vel[2];
                diag.forcing_l2 += du0*du0 + du1*du1 + du2*du2;
                fp->vel[0] = ur_block[0][ind];
                fp->vel[1] = ur_block[1][ind];
                fp->vel[2] = ur_block[2][ind];
                vloc[0] += ur_block[0][ind]*ur_block[0][ind];
                vloc[1] += ur_block[1][ind]*ur_block[1][ind];
                vloc[2] += ur_block[2][ind]*ur_block[2][ind];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, vloc, 3, MPI_DOUBLE, MPI_SUM, comm);
    f3d.fill_boundary_cc(false);
    if((forcing_calls % cfl_recheck_stride) == 0) update_dt_from_cfl();
    diag.t_unpack = MPI_Wtime()-t0;

    if(do_diag_sample) diag.div_after = l2_divergence();
    else diag.div_after = 0.0;

    // energies in physical space
    double E_local = 0.0;
    ind=0;
    for(int kk=0; kk<f3d.so; ++kk){
        for(int jj=0; jj<f3d.sn; ++jj){
            for(int ii=0; ii<f3d.sm; ++ii, ++ind){
                E_local += ur_block[0][ind]*ur_block[0][ind] + ur_block[1][ind]*ur_block[1][ind] + ur_block[2][ind]*ur_block[2][ind];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &E_local, 1, MPI_DOUBLE, MPI_SUM, comm);
    diag.E_total = 0.5*E_local/static_cast<double>(nx*ny*nz);
    diag.E_low = El;
    diag.E_high = Eh;
    diag.alpha = alpha;
    diag.target_err = diag.E_total - target_energy;
    diag.var_x = vloc[0]/(nx*ny*nz);
    diag.var_y = vloc[1]/(nx*ny*nz);
    diag.var_z = vloc[2]/(nx*ny*nz);
    diag.forcing_l2 = sqrt(diag.forcing_l2/(f3d.sm*f3d.sn*f3d.so + 1e-12));
    diag.eps = eps_spec;
    diag.re_lambda = (diag.eps>1e-20)?sqrt(20.0*diag.E_total*diag.E_total/(3.0*spars->fmu*diag.eps)):0.0;
}

void hit_module::compute_shell_spectrum(std::vector<double> &ek) {
    int nb = static_cast<int>(sqrt((nx/2.0)*(nx/2.0)+(ny/2.0)*(ny/2.0)+(nz/2.0)*(nz/2.0))/sqrt(3.0))+1;
    ek.assign(nb, 0.0);
    std::vector<double> cnt(nb, 0.0);
    ptrdiff_t nloc = local_n0 * static_cast<ptrdiff_t>(ny) * static_cast<ptrdiff_t>(nz);
    for(ptrdiff_t i=0;i<nloc;i++){
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
        fprintf(fh_energy, "%.12e %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                f3d.time, step, diag.E_total, diag.E_low, diag.E_high, diag.alpha, diag.target_err,
                diag.var_x, diag.var_y, diag.var_z, diag.forcing_l2, diag.div_before, diag.div_after,
                diag.eps, diag.re_lambda, diag.t_pack, diag.t_comm_g2s, diag.t_fft_fwd, diag.t_shell, diag.t_fft_bwd, diag.t_comm_s2g, diag.t_unpack);
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
    fwrite(&target_energy_dyn, sizeof(double), 1, fh);
    fwrite(&alpha_last, sizeof(double), 1, fh);
    fwrite(&eta, sizeof(double), 1, fh);
}

void hit_module::read_restart_state(FILE *fh) {
    if(!hit_enable) return;
    size_t ok = fread(&target_energy, sizeof(double), 1, fh);
    ok += fread(&target_energy_dyn, sizeof(double), 1, fh);
    ok += fread(&alpha_last, sizeof(double), 1, fh);
    ok += fread(&eta, sizeof(double), 1, fh);
    if(ok<4) p_fatal_error("hit_module::read_restart_state failed", 1);
    f3d.mgmt->set_insertion_eta(eta);
}

