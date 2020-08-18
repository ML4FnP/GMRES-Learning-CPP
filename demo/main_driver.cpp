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




// struct Lin_NetImpl : torch::nn::Module
// {
//   Lin_NetImpl(int64_t Dim ) : 
//         lin1(torch::nn::LinearOptions(Dim*Dim*Dim,Dim*Dim*Dim).bias(false))
//  {
//    // register_module() is needed if we want to use the parameters() method later on
//    register_module("lin1", lin1);
//  }

//  torch::Tensor forward(torch::Tensor x, int64_t Dim)
//  {

//    x = x.unsqueeze(0); // add batch dim (first index)
//    int64_t Current_batchsize= x.size(0);
//    x = x.reshape({Current_batchsize,1,1,-1}); // flatten
//    x = lin1(x);
//    x = x.reshape({Current_batchsize,Dim,Dim,Dim}); // unflatten
//    return x;
//  }

//  torch::nn::Linear lin1;

// };
// TORCH_MODULE(Lin_Net);




struct Net : torch::nn::Module {
  Net(int64_t Dim1, int64_t Dim2, int64_t Dim3,int64_t Dim4, int64_t Dim5, int64_t Dim6)
    : linear(register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(Dim1*Dim2*Dim3,Dim4*Dim5*Dim6).bias(false))))
  { }
   torch::Tensor forward(torch::Tensor x, int64_t Dim1, int64_t Dim2, int64_t Dim3,int64_t Dim4, int64_t Dim5, int64_t Dim6)
   {
    int64_t Current_batchsize= x.size(0);
    x = x.reshape({Current_batchsize,1,1,-1}); // flatten
    x = linear(x);
    x = x.reshape({Current_batchsize,Dim4,Dim5,Dim6}); // unflatten
    return x;
   }
  torch::nn::Linear linear;
};




class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        torch::Tensor bTensor, SolTensor;

    public:
        CustomDataset(torch::Tensor bIn, torch::Tensor SolIn)
        {
          bTensor = bIn;
          SolTensor = SolIn;
        };
        
        torch::data::Example<> get(size_t index) override
        {
          return {bTensor[index], SolTensor[index]};
        };

        torch::optional<size_t> size() const override
        {
          /* Return number of samples (This is not used, and can  return anything in this case)
            It appears that a size must be defined as done here (i.e this is not optional).
            The code breaks without this.
            see: https://discuss.pytorch.org/t/custom-dataloader/81874/3 */
          return SolTensor.size(0);
        };
  }; 






/* copy values of single box multifab to Pytorch Tensor  */
template<typename T_src>
void ConvertToTensor(const T_src & mf_in, torch::Tensor & tensor_out) {

    const BoxArray & ba            = mf_in.boxArray();
    const DistributionMapping & dm = mf_in.DistributionMap();
            int ncomp                = mf_in.nComp();
            int ngrow                = mf_in.nGrow();
            int i,j,k;

#ifdef _OPENMP
#pragma omp parallel
#endif

    for(MFIter mfi(mf_in, true); mfi.isValid(); ++ mfi) {
        const auto & in_tile  =      mf_in[mfi];

        for(BoxIterator bit(mfi.growntilebox()); bit.ok(); ++ bit)
         {
            i=int(bit()[0]) - int(in_tile.smallEnd()[0]);
            j=int(bit()[1]) - int(in_tile.smallEnd()[1]);
            k=int(bit()[2]) - int(in_tile.smallEnd()[2]);
            tensor_out.index({i,j,k}) = in_tile(bit());
        }
    }
}


/* copy values of  Pytorch Tensor to a single box multifab */
template<typename T_dest>
void TensorToMultifab(torch::Tensor tensor_in ,T_dest & mf_out) {


    int   i, j, k;
    const BoxArray & ba            = mf_out.boxArray();
    const DistributionMapping & dm = mf_out.DistributionMap();
            int ncomp                = mf_out.nComp();
            int ngrow                = mf_out.nGrow();
    double test ;


#ifdef _OPENMP
#pragma omp parallel
#endif

    for(MFIter mfi(mf_out, true); mfi.isValid(); ++ mfi) {
        auto & out_tile  =        mf_out[mfi];

        for(BoxIterator bit(mfi.growntilebox()); bit.ok(); ++ bit)
         {
             i = bit()[0]-out_tile.smallEnd()[0];
             j = bit()[1]-out_tile.smallEnd()[1];
             k = bit()[2]-out_tile.smallEnd()[2];
             out_tile(bit()) = tensor_in.index({i,j,k}).item<double>(); //RHS breaks when using iterator and smallEnd member function to compute index
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step )
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
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
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
{  
    return  dt;
}


torch::Tensor& Unpack_PresCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
{  
    return  PresCollect;
}

void CollectPressure(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step
                   ,torch::Tensor PresFinal)
{  

        PresCollect=torch::cat({PresCollect,PresFinal},0);
    
}

torch::Tensor& Unpack_RHSCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,torch::Tensor& RHSCollect
                   ,int step)
{  
    return  RHSCollect;
}


void CollectRHS(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, torch::Tensor& RHSCollect
                   ,int step,torch::Tensor RHSFinal)
{  

        RHSCollect=torch::cat({RHSCollect,RHSFinal},0);
    
}


int unpack_Step(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, torch::Tensor& RHSCollect
                   ,int step)
{  

       return step;
    
}




template<typename F>
auto Wrapper(F func,bool RefineSol ,torch::Device device,std::shared_ptr<Net> NETPres, const IntVect presTensordim, const IntVect srctermXTensordim)
{
    auto new_function = [func,RefineSol,device,NETPres,presTensordim,srctermXTensordim](auto&&... args)
    {
        
        func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
        Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
        Unpack_geom(args...),Unpack_dt(args...));

        int step=unpack_Step(args...);
        Print() << step << "\n";

        if(RefineSol == true and step>10)
        {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
                torch::Tensor presTensor= torch::zeros({presTensordim[0]+1 , presTensordim[1]+1,presTensordim[2]+1 },options);
                torch::Tensor RHSTensor= torch::zeros({srctermXTensordim[0]+1 , srctermXTensordim[1]+1,srctermXTensordim[2]+1 },options);
                ConvertToTensor(Unpack_pres(args...),presTensor);
                ConvertToTensor(Unpack_sourceTerms(args...)[0],RHSTensor);
                CollectPressure(args...,presTensor.unsqueeze(0));
                CollectRHS(args...,RHSTensor.unsqueeze(0));

                /*Setting up learning loop below */
                torch::optim::Adagrad optimizer(NETPres->parameters(), torch::optim::AdagradOptions(0.01));

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDataset(Unpack_RHSCollect(args...),Unpack_PresCollect(args...)).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 32;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 10000; 
                float loss = 10.0;

                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while(loss>e1 && epoch<numEpoch )
                {
                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 
                            // Solution data
                            torch::Tensor target = batch.target; 

                            // Reset gradients
                            optimizer.zero_grad();

                            // forward pass
                            torch::Tensor output = NETPres->forward(data.to(torch::kFloat32)
                                                ,int(srctermXTensordim[0]+1),int(srctermXTensordim[1]+1),int(srctermXTensordim[2]+1)
                                                ,int(presTensordim[0]+1),int(presTensordim[1]+1),int(presTensordim[2]+1));
                            //evaulate loss
                            // auto loss_out = torch::nn::functional::mse_loss(output,target, torch::nn::functional::MSELossFuncOptions(torch::kSum));
                            torch::Tensor loss_out = torch::mse_loss(output, target.to(torch::kFloat32));
                            loss = loss_out.item<float>();
                            // Backward pass
                            loss_out.backward();

                            // Apply gradients
                            optimizer.step();

                            // Print loop info to console
                            epoch = epoch +1;
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss: "  << loss << std::endl;
                    std::cout << "Epoch Number: " << epoch << std::endl;
                }



                // MultiFab pres ;
                // torch::Tensor output = NETPres->forward(presTensor,int(presTensordim[0]+1),int(presTensordim[1]+1),int(presTensordim[2]+1));
                // TensorToMultifab<amrex::MultiFab>(output,pres);
        }


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




// struct CNN_NetImpl : torch::nn::Module
// {
//   CNN_NetImpl(int64_t Dim ) : 
//         conv1(torch::nn::Conv3dOptions(1, 1, {15,15}).stride(1).padding({7,7}).bias(false)),
//         conv2(torch::nn::Conv3dOptions(1, 1, {13,13}).stride(1).padding({6,6}).bias(false)),
//         lin1(torch::nn::LinearOptions(Dim*Dim*Dim,Dim*Dim*Dim).bias(false)),
//         relu1(torch::nn::LeakyReLUOptions())
//  {
//    // register_module() is needed if we want to use the parameters() method later on
//    register_module("conv1", conv1);
//    register_module("conv2", conv2);
//    register_module("lin1", lin1);
//  }

//  torch::Tensor forward(torch::Tensor x, int64_t Dim)
//  {
//   int64_t Current_batchsize= x.size(0);
//    x = x.unsqueeze(1); // add channel dim (second index)
//    x = relu1(conv1(x));
//    x = relu1(conv2(x));
//    x = x.squeeze(1); // remove channel dim (second index)
//    x = x.reshape({Current_batchsize,1,1,-1}); // flatten
//    x = lin1(x);
//    x = x.reshape({Current_batchsize,Dim,Dim,Dim}); // unflatten
//    return x;
//    // std::cout << x.sizes() << std::endl;
//  }

//  torch::nn::Conv3d conv1,conv2;
//  torch::nn::Linear lin1;
//  torch::nn::LeakyReLU relu1;

// };
// TORCH_MODULE(CNN_Net);






















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


    //___________________________________________________
    // Setup ML 

    // Spread forces to RHS
    std::array<MultiFab, AMREX_SPACEDIM> source_terms;
    for (int d=0; d<AMREX_SPACEDIM; ++d){
        source_terms[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 6);
        source_terms[d].setVal(0.);
    }

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())  device = torch::Device(torch::kCUDA);

    const auto & presbox  =   pres[0];
    IntVect presTensordim = presbox.bigEnd()-presbox.smallEnd();

    const auto & sourceTermxbox  =   source_terms[0][0];
    IntVect srctermXTensordim = sourceTermxbox.bigEnd()-sourceTermxbox.smallEnd();
    


    //   Define model and move to GPU
    auto TestNet = std::make_shared<Net>(int(srctermXTensordim[0]+1),int(srctermXTensordim[1]+1),int(srctermXTensordim[2]+1)
                                        ,int(presTensordim[0]+1),int(presTensordim[1]+1),int(presTensordim[2]+1));
    TestNet->to(device);


    /* pointer  to advanceStokes functions in src_hydro/advance.cpp  */
    void (*advanceStokesPtr)(std::array< MultiFab, AMREX_SPACEDIM >&,MultiFab&, 
                                const std::array< MultiFab,AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&, 
                                MultiFab&,MultiFab&,std::array< MultiFab, NUM_EDGE >&,
                                const Geometry,const Real&
                                ) = &advanceStokes;

    /* Wrap advanceStokes function pointer */
    bool RefineSol=false;
    auto advanceStokes_ML=Wrapper(advanceStokesPtr,RefineSol,device,TestNet,presTensordim,srctermXTensordim) ;
    RefineSol=true;
    auto advanceStokes_ML2=Wrapper(advanceStokesPtr,RefineSol,device,TestNet,presTensordim,srctermXTensordim) ;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
    torch::Tensor presCollect= torch::zeros({1,presTensordim[0]+1, presTensordim[1]+1,presTensordim[2]+1},options);
    torch::Tensor RHSCollect= torch::zeros({1,srctermXTensordim[0]+1, srctermXTensordim[1]+1,srctermXTensordim[2]+1},options);

    /****************************************************************************
     *                                                                          *
     * Advance Time Steps                                                       *
     *                                                                          *
     ***************************************************************************/


    //___________________________________________________________________________

    while (step < 50)
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
        advanceStokes_ML(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt,presCollect,RHSCollect,step);
        // advanceStokes(
        //         umac, pres,              /* LHS */
        //         mfluxdiv, source_terms,  /* RHS */
        //         alpha_fc, beta, gamma, beta_ed, geom, dt
        //     );


        gmres::gmres_abs_tol = 1e-7;
        advanceStokes_ML2(umac,pres,mfluxdiv,source_terms,alpha_fc, beta, gamma, beta_ed, geom, dt,presCollect,RHSCollect,step);
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




    // // Call the timer again and compute the maximum difference between the start
    // // time and stop time over all processors
    // Real stop_time = ParallelDescriptor::second() - strt_time;
    // ParallelDescriptor::ReduceRealMax(stop_time);
    // amrex::Print() << "Run time = " << stop_time << std::endl;
}
