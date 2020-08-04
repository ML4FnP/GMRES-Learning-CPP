#include "main_driver.H"

#include "hydro_functions.H"

#include "StochMomFlux.H"

#include "common_functions.H"

#include "gmres_functions.H"

#include "common_namespace_declarations.H"

#include "gmres_namespace_declarations.H"

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

#include <IBMarkerContainer.H>

#include "MFUtil.H"

#include <torch/torch.h>

using namespace amrex;
using namespace torch::indexing;





/* copy values of single box multifab to Pytorch Tensor  */
template<typename T_src>
void ConvertToTensor(const T_src & mf_in, torch::Tensor & tensor_out) {

    const BoxArray & ba            = mf_in.boxArray();
    const DistributionMapping & dm = mf_in.DistributionMap();
            int ncomp                = mf_in.nComp();
            int ngrow                = mf_in.nGrow();

#ifdef _OPENMP
#pragma omp parallel
#endif

    for(MFIter mfi(mf_in, true); mfi.isValid(); ++ mfi) {
        const auto & in_tile  =      mf_in[mfi];

        for(BoxIterator bit(mfi.tilebox()); bit.ok(); ++ bit)
         {
            tensor_out.index({bit()[0],bit()[1],bit()[2]}) = in_tile(bit());
        }
    }
}


std::array< MultiFab, AMREX_SPACEDIM >& Unpack_umac(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  umac;
}

MultiFab& Unpack_pres(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  pres;
}


const std::array< MultiFab, AMREX_SPACEDIM >& Unpack_flux(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  stochMfluxdiv;
}


std::array< MultiFab, AMREX_SPACEDIM >& Unpack_sourceTerms(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  sourceTerms;
}


std::array< MultiFab, AMREX_SPACEDIM >& Unpack_alpha_fc(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  alpha_fc;
}

MultiFab& Unpack_beta(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  beta;
}

MultiFab& Unpack_gamma(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  gamma;
}


std::array< MultiFab, NUM_EDGE >& Unpack_beta_ed(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  beta_ed;
}

const Geometry Unpack_geom(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  geom;
}

const Real& Unpack_dt(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt)
{  
    return  dt;
}




template<typename F,typename R, typename U>
auto Wrapper(F func,R param1, U param2)
{
    auto new_function = [func,param1,param2](auto&&... args)
    {
        
        std::cout << "BEGIN decorating...\n"; 
        std::cout << "stuff before wrapee call. "<< "Param1 = "<< param1  << std::endl;
        // func(args...);   
        func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...));
        std::cout << "stuff after wrapee call.  "<< "Param2 = "<< param2  << std::endl;
        std::cout << "END decorating\n";

    };
    return new_function;
}






//! Defines staggered MultiFab arrays (BoxArrays set according to the
//! nodal_flag_[x,y,z]). Each MultiFab has 1 component, and 1 ghost cell
inline void defineFC(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                     const BoxArray & ba, const DistributionMapping & dm,
                     int nghost) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].define(convert(ba, nodal_flag_dir[i]), dm, 1, nghost);
}

inline void defineEdge(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                     const BoxArray & ba, const DistributionMapping & dm,
                     int nghost) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].define(convert(ba, nodal_flag_edge[i]), dm, 1, nghost);
}


//! Sets the value for each component of staggered MultiFab
inline void setVal(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                   Real set_val) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].setVal(set_val);
}




struct CNN_NetImpl : torch::nn::Module
{
  CNN_NetImpl(int64_t Dim ) : 
        conv1(torch::nn::Conv3dOptions(1, 1, {15,15}).stride(1).padding({7,7}).bias(false)),
        conv2(torch::nn::Conv3dOptions(1, 1, {13,13}).stride(1).padding({6,6}).bias(false)),
        lin1(torch::nn::LinearOptions(Dim*Dim*Dim,Dim*Dim*Dim).bias(false)),
        relu1(torch::nn::LeakyReLUOptions())
 {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("lin1", lin1);
 }

 torch::Tensor forward(torch::Tensor x, int64_t Dim)
 {
  int64_t Current_batchsize= x.size(0);
   x = x.unsqueeze(1); // add channel dim (second index)
   x = relu1(conv1(x));
   x = relu1(conv2(x));
   x = x.squeeze(1); // remove channel dim (second index)
   x = x.reshape({Current_batchsize,1,1,-1}); // flatten
   x = lin1(x);
   x = x.reshape({Current_batchsize,Dim,Dim,Dim}); // unflatten
   return x;
   // std::cout << x.sizes() << std::endl;
 }

 torch::nn::Conv3d conv1,conv2;
 torch::nn::Linear lin1;
 torch::nn::LeakyReLU relu1;

};
TORCH_MODULE(CNN_Net);

























// argv contains the name of the inputs file entered at the command line
void main_driver(const char * argv) {
    BL_PROFILE_VAR("main_driver()",main_driver);


    /****************************************************************************
     *                                                                          *
     * Initialize simulation                                                    *
     *                                                                          *
     ***************************************************************************/

    // store the current time so we can later compute total run time.
    Real strt_time = ParallelDescriptor::second();


    //___________________________________________________________________________
    // Load parameters from inputs file, and initialize global parameters
    std::string inputs_file = argv;

    // read in parameters from inputs file into F90 modules NOTE: we use "+1"
    // because of amrex_string_c_to_f expects a null char termination
    read_common_namelist(inputs_file.c_str(), inputs_file.size()+1);
    read_gmres_namelist(inputs_file.c_str(), inputs_file.size()+1);

    // copy contents of F90 modules to C++ namespaces NOTE: any changes to
    // global settings in fortran/c++ after this point need to be synchronized
    InitializeCommonNamespace();
    InitializeGmresNamespace();


    //___________________________________________________________________________
    // Set boundary conditions

    // is the problem periodic? set to 0 (not periodic) by default
    Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    for (int i=0; i<AMREX_SPACEDIM; ++i)
        if (bc_vel_lo[i] <= -1 && bc_vel_hi[i] <= -1)
            is_periodic[i] = 1;


    //___________________________________________________________________________
    // Make BoxArray, DistributionMapping, and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(           0,            0,            0));
        IntVect dom_hi(AMREX_D_DECL(n_cells[0]-1, n_cells[1]-1, n_cells[2]-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size"
        // along a direction note we are converting "Vector<int> max_grid_size"
        // to an IntVect
        ba.maxSize(IntVect(max_grid_size));

        // This defines the physical box, [-1, 1] in each direction
        RealBox real_box({AMREX_D_DECL(prob_lo[0], prob_lo[1], prob_lo[2])},
                         {AMREX_D_DECL(prob_hi[0], prob_hi[1], prob_hi[2])});

        // This defines a Geometry object
        geom.define(domain, & real_box, CoordSys::cartesian, is_periodic.data());
    }

    // how boxes are distrubuted among MPI processes
    DistributionMapping dmap(ba);


    //___________________________________________________________________________
    // Cell size, and time step
    Real dt         = fixed_dt;
    Real dtinv      = 1.0 / dt;
    const Real * dx = geom.CellSize();


    //___________________________________________________________________________
    // Initialize random number generators
    const int n_rngs = 1;

    // this seems really random :P
    int fhdSeed      = 1;
    int particleSeed = 2;
    int selectorSeed = 3;
    int thetaSeed    = 4;
    int phiSeed      = 5;
    int generalSeed  = 6;

    // each CPU gets a different random seed
    const int proc = ParallelDescriptor::MyProc();
    fhdSeed      += proc;
    particleSeed += proc;
    selectorSeed += proc;
    thetaSeed    += proc;
    phiSeed      += proc;
    generalSeed  += proc;

    // initialize rngs
    rng_initialize(
            & fhdSeed, & particleSeed, & selectorSeed, & thetaSeed, & phiSeed,
            & generalSeed
        );

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    /****************************************************************************
     *                                                                          *
     * Initialize physical parameters                                           *
     *                                                                          *
     ***************************************************************************/

    //___________________________________________________________________________
    // Set rho, alpha, beta, gamma:

    // rho is cell-centered
    MultiFab rho(ba, dmap, 1, 1);
    rho.setVal(1.);

    // alpha_fc is face-centered
    std::array< MultiFab, AMREX_SPACEDIM > alpha_fc;
    defineFC(alpha_fc, ba, dmap, 1);
    setVal(alpha_fc, dtinv);

    // beta is cell-centered
    MultiFab beta(ba, dmap, 1, 1);
    beta.setVal(visc_coef);

    // beta is on nodes in 2D, and is on edges in 3D
    std::array< MultiFab, NUM_EDGE > beta_ed;
#if (AMREX_SPACEDIM == 2)
    beta_ed[0].define(convert(ba, nodal_flag), dmap, 1, 1);
    beta_ed[0].setVal(visc_coef);
#elif (AMREX_SPACEDIM == 3)
    defineEdge(beta_ed, ba, dmap, 1);
    setVal(beta_ed, visc_coef);
#endif

    // cell-centered gamma
    MultiFab gamma(ba, dmap, 1, 1);
    gamma.setVal(0.);


    //___________________________________________________________________________
    // Define & initialize eta & temperature MultiFabs

    // eta & temperature
    const Real eta_const  = visc_coef;
    const Real temp_const = T_init[0];      // [units: K]


    // NOTE: eta and temperature live on both cell-centers and edges

    // eta & temperature cell centered
    MultiFab  eta_cc(ba, dmap, 1, 1);
    MultiFab temp_cc(ba, dmap, 1, 1);
    // eta & temperature nodal
    std::array< MultiFab, NUM_EDGE >   eta_ed;
    std::array< MultiFab, NUM_EDGE >  temp_ed;

    // eta_ed and temp_ed are on nodes in 2D, and on edges in 3D
#if (AMREX_SPACEDIM == 2)
    eta_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);
    temp_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);

    eta_ed[0].setVal(eta_const);
    temp_ed[0].setVal(temp_const);
#elif (AMREX_SPACEDIM == 3)
    defineEdge(eta_ed, ba, dmap, 1);
    defineEdge(temp_ed, ba, dmap, 1);

    setVal(eta_ed, eta_const);
    setVal(temp_ed, temp_const);
#endif

    // eta_cc and temp_cc are always cell-centered
    eta_cc.setVal(eta_const);
    temp_cc.setVal(temp_const);


    //___________________________________________________________________________
    // Define random fluxes mflux (momentum-flux) divergence, staggered in x,y,z

    // mfluxdiv predictor multifabs
    std::array< MultiFab, AMREX_SPACEDIM >  mfluxdiv;
    defineFC(mfluxdiv, ba, dmap, 1);
    setVal(mfluxdiv, 0.);

    Vector< amrex::Real > weights;
    // weights = {std::sqrt(0.5), std::sqrt(0.5)};
    weights = {1.0};


    //___________________________________________________________________________
    // Define velocities and pressure

    // pressure for GMRES solve
    MultiFab pres(ba, dmap, 1, 1); 
    pres.setVal(0.);  // initial guess

    // staggered velocities
    std::array< MultiFab, AMREX_SPACEDIM > umac;
    defineFC(umac, ba, dmap, 1);
    setVal(umac, 0.);



    /****************************************************************************
     *                                                                          *
     * Set Initial Conditions                                                   *
     *                                                                          *
     ***************************************************************************/

    //___________________________________________________________________________
    // Initialize immers boundary markers (they will be the force sources)
    // Make sure that the nghost (last argument) is big enough!
    IBMarkerContainer ib_mc(geom, dmap, ba, 10);

    int ib_lev = 0;


    Vector<RealVect> marker_positions(1);
    marker_positions[0] = RealVect{0.5,  0.5, 0.5};

    Vector<Real> marker_radii(1);
    marker_radii[0] = {0.02};

    int ib_label = 0; //need to fix for multiple dumbbells
    ib_mc.InitList(ib_lev, marker_radii, marker_positions, ib_label);

    ib_mc.fillNeighbors();
    ib_mc.PrintMarkerData(ib_lev);


    //___________________________________________________________________________
    // Ensure that ICs satisfy BCs

    pres.FillBoundary(geom.periodicity());
    MultiFabPhysBC(pres, geom, 0, 1, 0);

    for (int i=0; i<AMREX_SPACEDIM; i++) {
        umac[i].FillBoundary(geom.periodicity());
        MultiFabPhysBCDomainVel(umac[i], geom, i);
        MultiFabPhysBCMacVel(umac[i], geom, i);
    }


    // //___________________________________________________________________________
    // // Add random momentum fluctuations

    // // Declare object of StochMomFlux class
    // //StochMomFlux sMflux (ba, dmap, geom, n_rngs);

    // // Add initial equilibrium fluctuations
    // addMomFluctuations(umac, rho, temp_cc, initial_variance_mom, geom);

    // // Project umac onto divergence free field
    // MultiFab macphi(ba,dmap, 1, 1);
    // MultiFab macrhs(ba,dmap, 1, 1);
    // macrhs.setVal(0.);
    // MacProj_hydro(umac, rho, geom, true); // from MacProj_hydro.cpp

    int step = 0;
    Real time = 0.;


    //___________________________________________________________________________
    // Write out initial state
    WritePlotFile(step, time, geom, umac, pres, ib_mc);



    /* pointer  to advanceStokes functions in src_hydro/advance.cpp  */
    void (*advanceStokesPtr)(std::array< MultiFab, AMREX_SPACEDIM >&,MultiFab&, 
                                const std::array< MultiFab,AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&, 
                                MultiFab&,MultiFab&,std::array< MultiFab, NUM_EDGE >&,
                                const Geometry,const Real&
                                ) = &advanceStokes;

    /* Wrap advanceStokes function pointer */
    auto advanceStokes_ML=Wrapper(advanceStokesPtr, "Test output1",3.14159) ;//, "parameter 1", 3.14159);
    auto advanceStokes_ML2=Wrapper(advanceStokesPtr, "Test output2",3.14159) ;//, "parameter 1", 3.14159);

    /****************************************************************************
     *                                                                          *
     * Advance Time Steps                                                       *
     *                                                                          *
     ***************************************************************************/


    //___________________________________________________________________________

    while (step < 1)
    {
        // Spread forces to RHS
        std::array<MultiFab, AMREX_SPACEDIM> source_terms;
        for (int d=0; d<AMREX_SPACEDIM; ++d){
            source_terms[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 6);
            source_terms[d].setVal(0.);
        }


        RealVect f_0 = RealVect{2.0*dis(gen), 0.0, 0.0};
        // Print() << dis(gen) << "rng check" << "\n";

        for (IBMarIter pti(ib_mc, ib_lev); pti.isValid(); ++pti) {

            // Get marker data (local to current thread)
            TileIndex index(pti.index(), pti.LocalTileIndex());
            AoS & markers = ib_mc.GetParticles(ib_lev).at(index).GetArrayOfStructs();
            long np = ib_mc.GetParticles(ib_lev).at(index).numParticles();

            for (int i =0; i<np; ++i) {
                ParticleType & mark = markers[i];
                for (int d=0; d<AMREX_SPACEDIM; ++d)
                    mark.rdata(IBMReal::forcex + d) = f_0[d];
            }
        }
    

        // Spread to the `fc_force` multifab
        ib_mc.SpreadMarkers(0, source_terms);
        for (int d=0; d<AMREX_SPACEDIM; ++d)
            source_terms[d].SumBoundary(geom.periodicity());

        Real step_strt_time = ParallelDescriptor::second();

        // if(variance_coef_mom != 0.0) {

        //     //___________________________________________________________________
        //     // Fill stochastic terms

        //     sMflux.fillMomStochastic();

        //     // Compute stochastic force terms (and apply to mfluxdiv_*)
        //     sMflux.StochMomFluxDiv(mfluxdiv_predict, 0, eta_cc, eta_ed, temp_cc, temp_ed, weights, dt);
        //     sMflux.StochMomFluxDiv(mfluxdiv_correct, 0, eta_cc, eta_ed, temp_cc, temp_ed, weights, dt);
        // }

        // Example of overwriting the settings from inputs file

        // void (*advanceStokesPtr)(std::array< MultiFab, AMREX_SPACEDIM >&,  MultiFab&, const std::array< MultiFab, 
        //                             AMREX_SPACEDIM >&,std::array< MultiFab, AMREX_SPACEDIM >&,std::array< MultiFab, AMREX_SPACEDIM >&,
        //                             MultiFab&,MultiFab&,std::array< MultiFab, NUM_EDGE >&,const Geometry,const Real& ) = &advanceStokes;
        // auto advanceStokes_ML=Wrapper(advanceStokes, "parameter 1",3.14159) ;//, "parameter 1", 3.14159);


        gmres::gmres_abs_tol = 1e-4;
        advanceStokes_ML(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt);
        // advanceStokes(
        //         umac, pres,              /* LHS */
        //         mfluxdiv, source_terms,  /* RHS */
        //         alpha_fc, beta, gamma, beta_ed, geom, dt
        //     );


        gmres::gmres_abs_tol = 1e-7;
        advanceStokes_ML2(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt);
        // advanceStokes(
        //         umac, pres,              /* LHS */
        //         mfluxdiv, source_terms,  /* RHS */
        //         alpha_fc, beta, gamma, beta_ed, geom, dt
        //     );


        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        time = time + dt;
        step ++;
        // write out umac & pres to a plotfile
        WritePlotFile(step, time, geom, umac, pres, ib_mc);
    }


    // gmres::gmres_abs_tol = 1e-8;
    // advanceStokes_ML(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt);
    // gmres::gmres_abs_tol = 1e-9;
    // advanceStokes_ML2(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt);
    






    // code for converting pressure multifab to tensor
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())  device = torch::Device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA).requires_grad(false); 

    torch::Tensor presTensor= torch::ones({max_grid_size[0], max_grid_size[1],max_grid_size[2]},options);
    ConvertToTensor<amrex::MultiFab>(pres,presTensor);





    // // Call the timer again and compute the maximum difference between the start
    // // time and stop time over all processors
    // Real stop_time = ParallelDescriptor::second() - strt_time;
    // ParallelDescriptor::ReduceRealMax(stop_time);
    // amrex::Print() << "Run time = " << stop_time << std::endl;
}
