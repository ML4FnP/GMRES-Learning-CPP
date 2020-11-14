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


#include <ostream>

using namespace amrex;
using namespace torch::indexing;

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




/* Ideal to explicitly use std::shared_ptr<MyModule> rahter than a "TorchModule" */
/* Need to set up NN using this approach where the  module is registered and constructed in the initializer list  */
/* Can also instead first construct the holder with a null pointer and then assign it in the constructor */

/* Linear network */
struct Net : torch::nn::Module {
  Net(int64_t DimInFlat, int64_t DimOutFlat)
    :   linear(torch::nn::LinearOptions(DimInFlat,DimOutFlat).bias(false))
        // conv1(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("linear", linear);
        // register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x, const std::vector<int> SrcTermDims, const amrex::IntVect presTensordim,const std::vector<int> umacTensordims )
   {
        int64_t Current_batchsize= x.size(0);
        x = linear(x);
        return x;
   }
  torch::nn::Linear linear;
//   torch::nn::Conv3d conv1;
};

//////////////////////////////////////////////////////////////////////////////
/* Stokes CNN */
struct StokesCNNet_11 : torch::nn::Module {
  StokesCNNet_11(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    //,
    //    conv2(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(3).bias(false).padding_mode(torch::kZeros)),
    //    conv3(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(3).bias(false).padding_mode(torch::kZeros)),
    //    conv4(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(3).bias(false).padding_mode(torch::kZeros)),
    //    conv5(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(3).bias(false).padding_mode(torch::kZeros)),
    //    conv6(torch::nn::Conv3dOptions(3, 3, 1).stride(1).padding(0).dilation(1).groups(3).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
        // register_module("conv2", conv1);
        // register_module("conv3", conv1);
        // register_module("conv4", conv1);
        // register_module("conv5", conv1);
        // register_module("conv6", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        // x = linear(x);
        //    x = torch::relu(batch_norm1(conv1(x)));
        //    x = torch::relu(batch_norm2(conv2(x)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_12 : torch::nn::Module {
  StokesCNNet_12(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_13 : torch::nn::Module {
  StokesCNNet_13(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};


struct StokesCNNet_21 : torch::nn::Module {
  StokesCNNet_21(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_22 : torch::nn::Module {
  StokesCNNet_22(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_23 : torch::nn::Module {
  StokesCNNet_23(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};


struct StokesCNNet_31 : torch::nn::Module {
  StokesCNNet_31(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_32 : torch::nn::Module {
  StokesCNNet_32(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_33 : torch::nn::Module {
  StokesCNNet_33(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};


struct StokesCNNet_P1 : torch::nn::Module {
  StokesCNNet_P1(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_P2 : torch::nn::Module {
  StokesCNNet_P2(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};

struct StokesCNNet_P3 : torch::nn::Module {
  StokesCNNet_P3(int64_t DimInFlat, int64_t DimOutFlat)
    :  conv1(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(1).dilation(1).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("conv1", conv1);
    }

   torch::Tensor forward(torch::Tensor x)
   {
        int64_t Current_batchsize= x.size(0);
        x = torch::relu(conv1(x.unsqueeze(1)));
        return x.squeeze(1);
   }
   torch::nn::Conv3d conv1; //,conv2,conv3,conv4,conv5,conv6;
};


//////////////////////////////////////////////////////////////////////////////

/* Need to define a class to use the Pytorch data loader */
/* Dataset class for linear network */
class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        torch::Tensor bTensor, SolTensor;
        torch::Tensor SrcTensorFlatx,SrcTensorFlaty,SrcTensorFlatz,SrcTensorFlat;
        torch::Tensor umacFlatx,umacFlaty,umacFlatz,umacFlat;
        torch::Tensor bp,PressureFlat;


        torch::Tensor SrcTensorCat,Pres,umacTensorsCat,bTest,SrcTest;

    public:
        CustomDataset(std::array<torch::Tensor,AMREX_SPACEDIM> bIn, torch::Tensor SolIn,std::array<torch::Tensor,AMREX_SPACEDIM> umacTensors)
        {
          auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
          int64_t Current_batchsize= bIn[0].size(0);

          /* xp,xu */
          umacFlatx=umacTensors[0].reshape({Current_batchsize,1,1,-1});
          umacFlaty=umacTensors[1].reshape({Current_batchsize,1,1,-1});
          umacFlatz=umacTensors[2].reshape({Current_batchsize,1,1,-1});
          umacFlat=torch::cat({umacFlatx,umacFlaty},-1);
          umacFlat=torch::cat({umacFlat,umacFlatz},-1);
          PressureFlat=SolIn.reshape({Current_batchsize,1,1,-1});
          umacFlat=torch::cat({umacFlat,PressureFlat},-1);

        /* bu,bp */
          SrcTensorFlatx=bIn[0].reshape({Current_batchsize,1,1,-1});
          SrcTensorFlaty=bIn[1].reshape({Current_batchsize,1,1,-1});
          SrcTensorFlatz=bIn[2].reshape({Current_batchsize,1,1,-1});
          SrcTensorFlat=torch::cat({SrcTensorFlatx,SrcTensorFlaty},-1);
          SrcTensorFlat=torch::cat({SrcTensorFlat,SrcTensorFlatz},-1);
          bp =torch::zeros({Current_batchsize,1,1,PressureFlat.size(-1)},options);
          SrcTensorFlat=torch::cat({SrcTensorFlat,bp},-1);


          bTensor =SrcTensorFlat;
          SolTensor =umacFlat;

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

/* Dataset class for CNN  */
class CustomDatasetCNN : public torch::data::Dataset<CustomDatasetCNN>
{
    private:
        torch::Tensor bTensor, SolTensor;
        torch::Tensor SrcTensorCat,Pres,umacTensorsCat;

    public:
        CustomDatasetCNN(std::array<torch::Tensor,AMREX_SPACEDIM> SrcTensor, torch::Tensor Pres,
                            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensors,
                            const IntVect presTensordim, const std::vector<int> srctermTensordim, 
                            const std::vector<int> umacTensordims)
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
            int64_t Current_batchsize= SrcTensor[0].size(0);

            /* Add channel dim to ever tensor component of the Std::array<torch::tensor,Amrex_dim> objects */
            for (int d=0;d<AMREX_SPACEDIM;d++) 
            {
            SrcTensor[d]=SrcTensor[d].unsqueeze(1);
            umacTensors[d]=umacTensors[d].unsqueeze(1);
            }
            Pres=Pres.unsqueeze(1);

        

            // The tensors are padded so that every component is the same size as the largest component (currently N+3).
            // Note: This allows the tensors to be concatencated, and allows the most direct use of the pytorch dataloader
            // Note: the tensor is padded with a large value at the end of every dimension so it's easy to verfiy this is removed in our network

            int maxP   = *std::max_element(presTensordim.begin(), presTensordim.end());
            int maxU   = *std::max_element(umacTensordims.begin(), umacTensordims.end());
            int maxSrc = *std::max_element(srctermTensordim.begin(), srctermTensordim.end());
            int max1   = std::max(maxP,maxU);
            int maxDim    = std::max(max1,maxSrc);
            
            SrcTensor[0]= torch::nn::functional::pad(SrcTensor[0], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(10000000));
            SrcTensor[1]= torch::nn::functional::pad(SrcTensor[1], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(10000000));
            SrcTensor[2]= torch::nn::functional::pad(SrcTensor[2], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(10000000));

            umacTensors[0]= torch::nn::functional::pad(umacTensors[0], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[2], 0, maxDim-umacTensordims[1],0, maxDim-umacTensordims[0]}).mode(torch::kConstant).value(10000000));
            umacTensors[1]= torch::nn::functional::pad(umacTensors[1], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[5], 0, maxDim-umacTensordims[4],0, maxDim-umacTensordims[3]}).mode(torch::kConstant).value(10000000));
            umacTensors[2]= torch::nn::functional::pad(umacTensors[2], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[8], 0, maxDim-umacTensordims[7],0, maxDim-umacTensordims[6]}).mode(torch::kConstant).value(10000000));

            Pres= torch::nn::functional::pad(Pres, torch::nn::functional::PadFuncOptions({0, maxDim-presTensordim[2], 0, maxDim-presTensordim[1],0,maxDim-presTensordim[0]}).mode(torch::kConstant).value(10000000));


            /* Concatenate every tensor along the channel dim to yield a tensor of the form (N,3,) */
            SrcTensorCat  =torch::cat({SrcTensor[0],SrcTensor[1],SrcTensor[2]},1);
            umacTensorsCat=torch::cat({umacTensors[0],umacTensors[1],umacTensors[2]},1);

            /* Network input tensor */
            bTensor=SrcTensorCat;

            /* Network solution tensor */
            SolTensor=torch::cat({umacTensorsCat,Pres},1);    

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
void ConvertToTensor(const T_src & mf_in, torch::Tensor & tensor_out)
 {

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
                tensor_out.index({0,i,j,k}) = in_tile(bit());
            }
        }
}


/* copy values of  Pytorch Tensor to a single box multifab */
template<typename T_dest>
void TensorToMultifab(torch::Tensor tensor_in ,T_dest & mf_out) 
{
        int   i, j, k;
        const BoxArray & ba            = mf_out.boxArray();
        const DistributionMapping & dm = mf_out.DistributionMap();
                int ncomp                = mf_out.nComp();
                int ngrow                = mf_out.nGrow();


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
                out_tile(bit()) = tensor_in.index({0,i,j,k}).item<double>();
            }
        }
}


/* copy values of single box std:Array MultiFab to Pytorch Tensor  */
void Convert_StdArrMF_To_StdArrTensor(std::array< MultiFab, AMREX_SPACEDIM >& StdArrMF, std::array<torch::Tensor,AMREX_SPACEDIM> & tensor_out)
{

    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        ConvertToTensor(StdArrMF[d],tensor_out[d]);
    }
}


/* copy values of  std::array of Pytorch Tensors to std::array of single box multifabs */
void stdArrTensorTostdArrMultifab( std::array<torch::Tensor,AMREX_SPACEDIM>& tensor_in ,std::array< MultiFab, AMREX_SPACEDIM >& mf_out) 
{
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        TensorToMultifab(tensor_in[d],mf_out[d]);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* residual computation */

void ResidCompute (std::array< MultiFab, AMREX_SPACEDIM >& umac,
                   MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   Real& norm_resid)
{

    Real theta_alpha = 0.;
    Real norm_pre_rhs;

    const BoxArray& ba = beta.boxArray();
    const DistributionMapping& dmap = beta.DistributionMap();

    

    // rhs_p GMRES solve
    MultiFab gmres_rhs_p(ba, dmap, 1, 0);
    gmres_rhs_p.setVal(0.);

    // rhs_u GMRES solve
    std::array< MultiFab, AMREX_SPACEDIM > gmres_rhs_u;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        gmres_rhs_u[d].define(convert(ba,nodal_flag_dir[d]), dmap, 1, 0);
        gmres_rhs_u[d].setVal(0.);
    }

    // add forcing to gmres_rhs_u
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Add(gmres_rhs_u[d], stochMfluxdiv[d], 0, 0, 1, 0);
        MultiFab::Add(gmres_rhs_u[d], sourceTerms[d], 0, 0, 1, 0);
    }


    // GMRES varibles needed to compute residual
    StagMGSolver StagSolver;
    Precon Pcon;
    std::array<MultiFab, AMREX_SPACEDIM> alphainv_fc;
    std::array< MultiFab, AMREX_SPACEDIM > tmp_u;
    std::array< MultiFab, AMREX_SPACEDIM > scr_u;
    std::array< MultiFab, AMREX_SPACEDIM > r_u;
    MultiFab tmp_p;
    MultiFab scr_p;
    MultiFab r_p;



    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        r_u[d]        .define(convert(ba, nodal_flag_dir[d]), dmap, 1,                 1);
        // w_u[d]        .define(convert(ba, nodal_flag_dir[d]), dmap, 1,                 0);
        tmp_u[d]      .define(convert(ba, nodal_flag_dir[d]), dmap, 1,                 0);
        scr_u[d]      .define(convert(ba, nodal_flag_dir[d]), dmap, 1,                 0);
        // V_u[d]        .define(convert(ba, nodal_flag_dir[d]), dmap, gmres_max_inner+1, 0);
        alphainv_fc[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 0);
    } 
    r_p.define  (ba, dmap,                  1, 1);
    // w_p.define  (ba_in, dmap_in,                  1, 0);
    tmp_p.define(ba, dmap,                  1, 0);
    scr_p.define(ba, dmap,                  1, 0);
    // V_p.define  (ba_in, dmap_in,gmres_max_inner + 1, 0); // Krylov vectors
    StagSolver.Define(ba,dmap,geom);
    Pcon.Define(ba,dmap,geom);


    // Vector<Real> cs(gmres_max_inner);
    // Vector<Real> sn(gmres_max_inner);
    // Vector<Real>  y(gmres_max_inner);
    // Vector<Real>  s(gmres_max_inner+1);

    // Vector<Vector<Real>> H(gmres_max_inner + 1, Vector<Real>(gmres_max_inner));

    // int outer_iter, total_iter, i_copy; // for looping iteration
    // int i=0;

    Real norm_b;            // |b|;           computed once at beginning
    Real norm_pre_b;        // |M^-1 b|;      computed once at beginning
    // Real norm_resid;        // |M^-1 (b-Ax)|; computed at beginning of each outer iteration
    // Real norm_init_resid;   // |M^-1 (b-Ax)|; computed once at beginning
    Real norm_resid_Stokes; // |b-Ax|;        computed at beginning of each outer iteration
    // Real norm_init_Stokes;  // |b-Ax|;        computed once at beginning
    Real norm_u_noprecon;   // u component of norm_resid_Stokes
    Real norm_p_noprecon;   // p component of norm_resid_Stokes
    // Real norm_resid_est;

    Real norm_u; // temporary norms used to build full-state norm
    Real norm_p; // temporary norms used to build full-state norm

    // Vector<Real> inner_prod_vel(AMREX_SPACEDIM);
    // Real inner_prod_pres;  

    // set alphainv_fc to 1/alpha_fc
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        alphainv_fc[d].setVal(1.);
        alphainv_fc[d].divide(alpha_fc[d],0,1,0);
    }

    if (scale_factor != 1.) {
        theta_alpha = theta_alpha*scale_factor;

        // we will solve for scale*x_p so we need to scale the initial guess
        pres.mult(scale_factor, 0, 1, pres.nGrow());

        // scale the rhs:
        for (int d=0; d<AMREX_SPACEDIM; ++d)
            gmres_rhs_u[d].mult(scale_factor,0,1,gmres_rhs_u[d].nGrow());

        // scale the viscosities:
        beta.mult(scale_factor, 0, 1, beta.nGrow());
        gamma.mult(scale_factor, 0, 1, gamma.nGrow());
        for (int d=0; d<NUM_EDGE; ++d)
            beta_ed[d].mult(scale_factor, 0, 1, beta_ed[d].nGrow());
    }

    // First application of preconditioner
    Pcon.Apply(gmres_rhs_u, gmres_rhs_p, tmp_u, tmp_p, alpha_fc, alphainv_fc,
               beta, beta_ed, gamma, theta_alpha, geom, StagSolver);

    // preconditioned norm_b: norm_pre_b
    StagL2Norm(geom, tmp_u, 0, scr_u, norm_u);
    CCL2Norm(tmp_p, 0, scr_p, norm_p);
    norm_p       = p_norm_weight*norm_p;
    norm_pre_b   = sqrt(norm_u*norm_u + norm_p*norm_p);
    norm_pre_rhs = norm_pre_b;

    // calculate the l2 norm of rhs
    StagL2Norm(geom, gmres_rhs_u, 0, scr_u, norm_u);
    CCL2Norm(gmres_rhs_p, 0, scr_p, norm_p);
    norm_p = p_norm_weight*norm_p;
    norm_b = sqrt(norm_u*norm_u + norm_p*norm_p);

    // Calculate tmp = Ax
    ApplyMatrix(tmp_u, tmp_p, umac, pres, alpha_fc, beta, beta_ed, gamma, theta_alpha, geom);

    // tmp = b - Ax
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Subtract(tmp_u[d], gmres_rhs_u[d], 0, 0, 1, 0);
        tmp_u[d].mult(-1., 0, 1, 0);
    }
    MultiFab::Subtract(tmp_p, gmres_rhs_p, 0, 0, 1, 0);
    tmp_p.mult(-1., 0, 1, 0);

    // un-preconditioned residuals
    StagL2Norm(geom, tmp_u, 0, scr_u, norm_u_noprecon);
    CCL2Norm(tmp_p, 0, scr_p, norm_p_noprecon);
    norm_p_noprecon   = p_norm_weight*norm_p_noprecon;
    norm_resid_Stokes = sqrt(norm_u_noprecon*norm_u_noprecon + norm_p_noprecon*norm_p_noprecon);

    // solve for r = M^{-1} tmp
    Pcon.Apply(tmp_u, tmp_p, r_u, r_p, alpha_fc, alphainv_fc,
                beta, beta_ed, gamma, theta_alpha, geom, StagSolver);


    // resid = sqrt(dot_product(r, r))
    StagL2Norm(geom, r_u, 0, scr_u, norm_u);
    CCL2Norm(r_p, 0, scr_p, norm_p);
    norm_p     = p_norm_weight*norm_p;
    norm_resid = sqrt(norm_u*norm_u + norm_p*norm_p);

    // amrex::Print() << norm_resid << " *************************************" << " \n";
    // amrex::Print() << norm_resid_Stokes << " *************************************" << " \n";

}


// gmres_rhs_u,gmres_rhs_p,umac,pres,
// bu,bp,xu,xp

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Functions for unpacking parameter pack passed to wrapped function. */


std::array< MultiFab, AMREX_SPACEDIM >& Unpack_umac(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step , std::vector<double>& TimeDataWindow, std::vector<double>& ResidDataWindow )
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  PresCollect;
}


std::array<torch::Tensor,AMREX_SPACEDIM>& Unpack_RHSCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  RHSCollect;
}


std::array<torch::Tensor,AMREX_SPACEDIM>& Unpack_umacCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  umacCollect;
}


int unpack_Step(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  

       return step;
    
}

std::vector<double> unpack_TimeDataWindow(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  

       return TimeDataWindow;
    
}




std::vector<double> unpack_ResidDataWindow(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  

       return TimeDataWindow;
    
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* functions that require something from the parameter pack ( or act on the pack) but do not simply unpack values are below */

void CollectPressure(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,torch::Tensor PresFinal)
{  

        PresCollect=torch::cat({PresCollect,PresFinal},0);
    
}

void CollectRHS(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::array<torch::Tensor,AMREX_SPACEDIM> RHSFinal)
{  
        for (int d=0; d<AMREX_SPACEDIM; ++d)
        {
            RHSCollect[d]=torch::cat({RHSCollect[d],RHSFinal[d]},0);
        }
}

void Collectumac(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::array<torch::Tensor,AMREX_SPACEDIM> umacFinal)
{  
        for (int d=0; d<AMREX_SPACEDIM; ++d)
        {
            umacCollect[d]=torch::cat({umacCollect[d],umacFinal[d]},0);
        }
}

void update_TimeDataWindow(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::vector<double>& TimeDataWindow_In)
{  
       TimeDataWindow=TimeDataWindow_In;
}


void update_ResidDataWindow(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::vector<double>& ResidDataWindow_In)
{  
       ResidDataWindow=ResidDataWindow_In;
}



 void TrimSourceMultiFab(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,amrex::DistributionMapping dmap, BoxArray  ba, std::array<MultiFab, AMREX_SPACEDIM>& source_termsTrimmed)
{  
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
        MultiFab::Copy(source_termsTrimmed[d], sourceTerms[d], 0, 0, 1, 1); 
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void  TrainLoop(std::shared_ptr<Net> NETPres,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,
                const IntVect presTensordim, const std::vector<int> srctermTensordim, 
                const std::vector<int> umacTensordims)
{
/*Setting up learning loop below */
                torch::optim::Adagrad optimizer(NETPres->parameters(), torch::optim::AdagradOptions(0.01));

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDataset(RHSCollect,PresCollect,umacCollect).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 16;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 250; 
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
                            torch::Tensor output = NETPres->forward(data.to(torch::kFloat32),srctermTensordim,presTensordim,umacTensordims);

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
                    // std::cout << "___________" << std::endl;
                    // std::cout << "Loss: "  << loss << std::endl;
                    // std::cout << "Epoch Number: " << epoch << std::endl;
                }

}


void  CNN_TrainLoop(std::shared_ptr<StokesCNNet_11> CNN_11,std::shared_ptr<StokesCNNet_12> CNN_12,std::shared_ptr<StokesCNNet_13> CNN_13,
                       std::shared_ptr<StokesCNNet_21> CNN_21,std::shared_ptr<StokesCNNet_22> CNN_22,std::shared_ptr<StokesCNNet_23> CNN_23,
                       std::shared_ptr<StokesCNNet_31> CNN_31,std::shared_ptr<StokesCNNet_32> CNN_32,std::shared_ptr<StokesCNNet_33> CNN_33,
                       std::shared_ptr<StokesCNNet_P1> CNN_P1,std::shared_ptr<StokesCNNet_P2> CNN_P2,std::shared_ptr<StokesCNNet_P3> CNN_P3,
                       std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,torch::Tensor& PresCollect,
                       std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,const IntVect presTensordim, 
                       const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
{
                /*Setting up learning loop below */

                torch::optim::Adagrad optimizerUx({CNN_11->parameters(),CNN_12->parameters(),CNN_13->parameters()}, torch::optim::AdagradOptions(0.01));
                torch::optim::Adagrad optimizerUy({CNN_21->parameters(),CNN_22->parameters(),CNN_23->parameters()}, torch::optim::AdagradOptions(0.01));
                torch::optim::Adagrad optimizerUz({CNN_31->parameters(),CNN_32->parameters(),CNN_33->parameters()}, torch::optim::AdagradOptions(0.01));
                torch::optim::Adagrad optimizerPres({CNN_P1->parameters(),CNN_P2->parameters(),CNN_P3->parameters()}, torch::optim::AdagradOptions(0.01));

                // torch::optim::SGD optimizerUx({CNN_11->parameters(),CNN_12->parameters(),CNN_13->parameters()}, torch::optim::SGDOptions (0.001));
                // torch::optim::SGD optimizerUy({CNN_21->parameters(),CNN_22->parameters(),CNN_23->parameters()}, torch::optim::SGDOptions (0.001));
                // torch::optim::SGD optimizerUz({CNN_31->parameters(),CNN_32->parameters(),CNN_33->parameters()}, torch::optim::SGDOptions (0.001));
                // torch::optim::SGD optimizerPres({CNN_P1->parameters(),CNN_P2->parameters(),CNN_P3->parameters()}, torch::optim::SGDOptions (0.001));

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN(RHSCollect,PresCollect,umacCollect,presTensordim, srctermTensordim,umacTensordims).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 16;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 250; 

                float lossUx = 10.0;
                float lossUy = 10.0;
                float lossUz = 10.0;
                float lossP  = 10.0;

                // Compute maximal dimension for slicing down tensor output of dataloader object
                int maxP   = *std::max_element(presTensordim.begin(), presTensordim.end());
                int maxU   = *std::max_element(umacTensordims.begin(), umacTensordims.end());
                int maxSrc = *std::max_element(srctermTensordim.begin(), srctermTensordim.end());
                int max1   = std::max(maxP,maxU);
                int maxDim    = std::max(max1,maxSrc);

               
                // Set indicies for removing the padded elements of the tensors used in the dataloader
                // Note: To use the pytorch dataloader(which makes it easy to use the autograd feature)
                // we must set tensor data pairs (target and data) as an output. 
                // To do this, the source,umac,and pressure Multifabs
                // that are converted to tensors are first concatenated by pading them 
                // so that every dimension has the same size as the largest dimension.
                // Below, tensor indicies are set so we remove this padding
                auto Ux_f1_Slice_x =Slice(0,srctermTensordim[0]-maxDim);
                auto Ux_f2_Slice_y =Slice(0,srctermTensordim[1]-maxDim);
                auto Ux_f3_Slice_z =Slice(0,srctermTensordim[2]-maxDim);
                auto Uy_f1_Slice_x =Slice(0,srctermTensordim[3]-maxDim);
                auto Uy_f2_Slice_y =Slice(0,srctermTensordim[4]-maxDim);
                auto Uy_f3_Slice_z =Slice(0,srctermTensordim[5]-maxDim);
                auto Uz_f1_Slice_x =Slice(0,srctermTensordim[6]-maxDim);
                auto Uz_f2_Slice_y =Slice(0,srctermTensordim[7]-maxDim);
                auto Uz_f3_Slice_z =Slice(0,srctermTensordim[8]-maxDim);

                if (srctermTensordim[0]-maxDim==0) Ux_f1_Slice_x =Slice();
                if (srctermTensordim[1]-maxDim==0) Ux_f2_Slice_y =Slice();
                if (srctermTensordim[2]-maxDim==0) Ux_f3_Slice_z =Slice();
                if (srctermTensordim[3]-maxDim==0) Uy_f1_Slice_x =Slice();
                if (srctermTensordim[4]-maxDim==0) Uy_f2_Slice_y =Slice();
                if (srctermTensordim[5]-maxDim==0) Uy_f3_Slice_z =Slice();
                if (srctermTensordim[6]-maxDim==0) Uz_f1_Slice_x =Slice();
                if (srctermTensordim[7]-maxDim==0) Uz_f2_Slice_y =Slice();
                if (srctermTensordim[8]-maxDim==0) Uz_f3_Slice_z =Slice();


                // auto Ux_f2_Slice_y =Slice(0, srctermTensordim[1]-maxDim);
                // auto Ux_f3_Slice_z =Slice(0, srctermTensordim[2]-maxDim);

                // auto Uy_f1_Slice_x =Slice(0, srctermTensordim[3]-maxDim);
                // auto Uy_f2_Slice_y =Slice();
                // auto Uy_f3_Slice_z =Slice(0, srctermTensordim[5]-maxDim);

                // auto Uz_f1_Slice_x =Slice(0, srctermTensordim[6]-maxDim);
                // auto Uz_f2_Slice_y =Slice(0, srctermTensordim[7]-maxDim);
                // auto Uz_f3_Slice_z =Slice();




                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while((lossUx+lossUy+lossUz+lossP)>5*e1 and epoch<numEpoch )
                {


                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 

                            // Solution data
                            torch::Tensor target = batch.target; 

   
                            // Reset gradients
                            optimizerUx.zero_grad();
                            optimizerUy.zero_grad();
                            optimizerUz.zero_grad();
                            optimizerPres.zero_grad();

                            // forward pass
                            // Note: Inputs to forward function are de-paded appropriately
                            torch::Tensor outputUx1 = CNN_11->forward(data.index({Slice(),0,Ux_f1_Slice_x,Ux_f2_Slice_y,Ux_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUx2 = CNN_12->forward(data.index({Slice(),1,Uy_f1_Slice_x,Uy_f2_Slice_y,Uy_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUx3 = CNN_13->forward(data.index({Slice(),2,Uz_f1_Slice_x,Uz_f2_Slice_y,Uz_f3_Slice_z}).to(torch::kFloat32));

                            torch::Tensor outputUy1 = CNN_21->forward(data.index({Slice(),0,Ux_f1_Slice_x,Ux_f2_Slice_y,Ux_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUy2 = CNN_22->forward(data.index({Slice(),1,Uy_f1_Slice_x,Uy_f2_Slice_y,Uy_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUy3 = CNN_23->forward(data.index({Slice(),2,Uz_f1_Slice_x,Uz_f2_Slice_y,Uz_f3_Slice_z}).to(torch::kFloat32));

                            torch::Tensor outputUz1 = CNN_31->forward(data.index({Slice(),0,Ux_f1_Slice_x,Ux_f2_Slice_y,Ux_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUz2 = CNN_32->forward(data.index({Slice(),1,Uy_f1_Slice_x,Uy_f2_Slice_y,Uy_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputUz3 = CNN_33->forward(data.index({Slice(),2,Uz_f1_Slice_x,Uz_f2_Slice_y,Uz_f3_Slice_z}).to(torch::kFloat32));

                            torch::Tensor outputP1 = CNN_P1->forward(data.index({Slice(),0,Ux_f1_Slice_x,Ux_f2_Slice_y,Ux_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputP2 = CNN_P2->forward(data.index({Slice(),1,Uy_f1_Slice_x,Uy_f2_Slice_y,Uy_f3_Slice_z}).to(torch::kFloat32));
                            torch::Tensor outputP3 = CNN_P3->forward(data.index({Slice(),2,Uz_f1_Slice_x,Uz_f2_Slice_y,Uz_f3_Slice_z}).to(torch::kFloat32));

                            //evaulate loss
                            // auto loss_out = torch::nn::functional::mse_loss(output,target, torch::nn::functional::MSELossFuncOptions(torch::kSum));
                            // Note: target solution data to  loss function are de-paded appropriately to match de-padded forward function inputs
                            torch::Tensor loss_outUx = torch::mse_loss(outputUx1, target.index({Slice(),0,Slice(),Slice(0,-1),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUx2, target.index({Slice(),1,Slice(0,-1),Slice(),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUx3, target.index({Slice(),2,Slice(0,-1),Slice(0,-1),Slice()}).to(torch::kFloat32));

                            torch::Tensor loss_outUy = torch::mse_loss(outputUy1, target.index({Slice(),0,Slice(),Slice(0,-1),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUy2, target.index({Slice(),1,Slice(0,-1),Slice(),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUy3, target.index({Slice(),2,Slice(0,-1),Slice(0,-1),Slice()}).to(torch::kFloat32));
                            
                            torch::Tensor loss_outUz = torch::mse_loss(outputUz1, target.index({Slice(),0,Slice(),Slice(0,-1),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUz2, target.index({Slice(),1,Slice(0,-1),Slice(),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputUz3, target.index({Slice(),2,Slice(0,-1),Slice(0,-1),Slice()}).to(torch::kFloat32));

                            torch::Tensor loss_outP  = torch::mse_loss(outputP1, target.index({Slice(),0,Slice(),Slice(0,-1),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputP2, target.index({Slice(),1,Slice(0,-1),Slice(),Slice(0,-1)}).to(torch::kFloat32))
                                                   + torch::mse_loss(outputP3, target.index({Slice(),2,Slice(0,-1),Slice(0,-1),Slice()}).to(torch::kFloat32));


                            lossUx = loss_outUx.item<float>();
                            lossUy = loss_outUy.item<float>();
                            lossUz = loss_outUz.item<float>();
                            lossP  = loss_outP.item<float>();

                            // Print()<< "TEST LOSS" << lossUx << " \n"; 

                            // Backward pass
                            loss_outUx.backward();
                            loss_outUy.backward();
                            loss_outUz.backward();
                            loss_outP.backward();

                            // Apply gradients
                            optimizerUx.step();
                            optimizerUy.step();
                            optimizerUz.step();
                            optimizerPres.step();

                            // Print loop info to console
                            epoch = epoch +1;
                    }
                    // std::cout << "___________" << std::endl;
                    // std::cout << "Loss Ux: "  << lossUx << std::endl;
                    // std::cout << "Loss Uy: "  << lossUy << std::endl;
                    // std::cout << "Loss Uz: "  << lossUz << std::endl;
                    // std::cout << "Loss P : "  << lossP << std::endl;

                    // std::cout << "Epoch Number: " << epoch << std::endl;
                }

}


double MovingAvg (std::vector<double>& InDataWindow)
{
    int window = InDataWindow.size();
    double sum;
    for(int i = 0 ; i<window ; i++)
    {
        sum+=InDataWindow[i];
    }
    sum = sum/double(window);
    return sum;
}




/* ML Wrapper for advanceStokes */
template<typename F>
auto Wrapper(F func,bool RefineSol ,torch::Device device,std::shared_ptr<Net> NETPres, const IntVect presTensordim, const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims,amrex::DistributionMapping dmap, BoxArray  ba)
{
    auto new_function = [func,RefineSol,device,NETPres,presTensordim,srctermTensordim,umacTensordims,dmap,ba](auto&&... args)
    {
        int retrainFreq =6;
        int initNum     =20;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
        int step=unpack_Step(args...);
        std::vector<double> TimeDataWindow =unpack_TimeDataWindow(args...);
        std::vector<double> ResidDataWindow =unpack_ResidDataWindow(args...);
        int WindowIdx=((step-initNum)-1)%TimeDataWindow.size();
        bool use_NN_prediction =false;



        MultiFab presNN(ba, dmap, 1, 1); 
        presNN.setVal(0.);  

        std::array< MultiFab, AMREX_SPACEDIM > umacNN;
        defineFC(umacNN, ba, dmap, 1);
        setVal(umacNN, 0.);



        /* Use NN to predict pressure */
        if (RefineSol==false and step>initNum )
        {
            std::array<torch::Tensor,AMREX_SPACEDIM> RHSTensor;
            RHSTensor[0]=torch::zeros({1,srctermTensordim[0] , srctermTensordim[1],srctermTensordim[2] },options);
            RHSTensor[1]=torch::zeros({1,srctermTensordim[3] , srctermTensordim[4],srctermTensordim[5] },options);
            RHSTensor[2]=torch::zeros({1,srctermTensordim[6] , srctermTensordim[7],srctermTensordim[8] },options);

            
            std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
            TrimSourceMultiFab(args...,dmap,ba,source_termsTrimmed);
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSTensor); /* Convert Std::array<MultiFab,AMREX_SPACEDIM > to  std::array<torch::tensor, AMREX_SPACEDIM> */




            /* Appropriately flatten input */ 
            torch::Tensor SrcTensorFlatx,SrcTensorFlaty,SrcTensorFlatz,SrcTensorFlat,PressureFlat,bp;

            torch::Tensor presTensorTemp= torch::zeros({presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            PressureFlat=presTensorTemp.reshape({1,1,-1});

            SrcTensorFlatx=RHSTensor[0].reshape({1,1,1,-1});
            SrcTensorFlaty=RHSTensor[1].reshape({1,1,1,-1});
            SrcTensorFlatz=RHSTensor[2].reshape({1,1,1,-1});
            SrcTensorFlat=torch::cat({SrcTensorFlatx,SrcTensorFlaty},-1);
            SrcTensorFlat=torch::cat({SrcTensorFlat,SrcTensorFlatz},-1);
            bp =torch::zeros({1,1,1,PressureFlat.size(-1)},options);
            SrcTensorFlat=torch::cat({SrcTensorFlat,bp},-1);

            /* Get prediction as tensor */
            torch::Tensor ModelOut = NETPres->forward(SrcTensorFlat,srctermTensordim,presTensordim,umacTensordims);


            /* Extract and reshape flattended pressure output */
            torch::Tensor presTensor = ModelOut.index({Slice(),Slice(),Slice(),Slice(-presTensordim[0]*presTensordim[1]*presTensordim[2]-1,-1)}); // Pickout pressure
            presTensor = presTensor.reshape({1,presTensordim[0],presTensordim[1],presTensordim[2]}); // unflatten


            /* Extract and reshape flattended staggered velocity output */
            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensor;
            int umacXFlatLength=umacTensordims[0]*umacTensordims[1]*umacTensordims[2];
            int umacYFlatLength=umacTensordims[3]*umacTensordims[4]*umacTensordims[5];
            int umacZFlatLength=umacTensordims[6]*umacTensordims[7]*umacTensordims[8];
            umacTensor[0] = ModelOut.index({Slice(),Slice(),Slice(),Slice(0,umacXFlatLength)}); // Pickout umacx
            umacTensor[1] = ModelOut.index({Slice(),Slice(),Slice(),Slice(umacXFlatLength,umacXFlatLength+umacYFlatLength)}); // Pickout umacy
            umacTensor[2] = ModelOut.index({Slice(),Slice(),Slice(),Slice(umacXFlatLength+umacYFlatLength,umacXFlatLength+umacYFlatLength+umacZFlatLength)}); // Pickout umacz
            umacTensor[0] = umacTensor[0].reshape({1,umacTensordims[0],umacTensordims[1],umacTensordims[2]}); // unflatten
            umacTensor[1] = umacTensor[1].reshape({1,umacTensordims[3],umacTensordims[4],umacTensordims[5]}); // unflatten
            umacTensor[2] = umacTensor[2].reshape({1,umacTensordims[6],umacTensordims[7],umacTensordims[8]}); // unflatten


            /* Convert tensor to multifab using distribution map of original pressure MultiFab */
            TensorToMultifab(presTensor,presNN);

            /* Convert std::array tensor to std::array multifab using distribution map of original pressure MultiFab */
            stdArrTensorTostdArrMultifab(umacTensor,umacNN);


            Real norm_residNN;
            Real norm_resid;

            gmres_max_iter = 10 ;

            func(umacNN,presNN,Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...));

            ResidCompute(umacNN,presNN,Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...),norm_residNN);

            func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...));

            ResidCompute(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...),norm_resid);

            gmres_max_iter = 100;

            // amrex::Print() <<  "Direct resid "<<  norm_resid << " *****************" << " \n";
            // amrex::Print() <<  "NN resid "<<  norm_residNN << " *****************" << " \n";

            if (norm_residNN < norm_resid)
            {
                amrex::Print() << "Use guess provided by NN" << " \n";
                use_NN_prediction=true;
            }else if(norm_residNN > norm_resid)
            {
                amrex::Print() << "10th GMRES resid for NN guess is too high. Discarding NN guess " << " \n";
                use_NN_prediction=false;
            }
            
            ResidDataWindow[WindowIdx]= norm_residNN; /* Add residual to window of values */
        }

        /* Evaluate wrapped function with either the NN prediction or original input */
        if(RefineSol==false and step>initNum and use_NN_prediction==true)
        {
            Real step_strt_time = ParallelDescriptor::second();

            func(umacNN,presNN,Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...));

            Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
            ParallelDescriptor::ReduceRealMax(step_stop_time);
            
            TimeDataWindow[WindowIdx]= step_stop_time; /* Add time to window of values */
            update_TimeDataWindow(args...,TimeDataWindow);

            std::ofstream outfile;
            outfile.open("TimeData.txt", std::ios_base::app); // append instead of overwrite
            outfile << step_stop_time << std::setw(10) << " \n"; 

        }else if(RefineSol==false and step>initNum and use_NN_prediction==false)
        {
            Real step_strt_time = ParallelDescriptor::second();

            func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...));

            Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
            ParallelDescriptor::ReduceRealMax(step_stop_time);
            
            TimeDataWindow[WindowIdx]= step_stop_time; /* Add time to window of values */
            update_TimeDataWindow(args...,TimeDataWindow);

            std::ofstream outfile;
            outfile.open("TimeData.txt", std::ios_base::app); // append instead of overwrite
            outfile << step_stop_time << std::setw(10) << " \n"; 

        }else if (RefineSol==false and step<initNum)
        {
            Real step_strt_time = ParallelDescriptor::second();

            func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
            Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
            Unpack_geom(args...),Unpack_dt(args...));  

            Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
            ParallelDescriptor::ReduceRealMax(step_stop_time);

            std::ofstream outfile;
            outfile.open("TimeData.txt", std::ios_base::app); // append instead of overwrite
            outfile << step_stop_time << std::setw(10) << " \n"; 

        }else
        {
            func(Unpack_umac(args...),Unpack_pres(args...),Unpack_flux(args...),Unpack_sourceTerms(args...),
            Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
            Unpack_geom(args...),Unpack_dt(args...));  
        }






        /* Add data to collection of tensors*/
        if(RefineSol == true and step<(initNum+TimeDataWindow.size()))
        {

            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);


            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensor;
            umacTensor[0]=torch::zeros({1,umacTensordims[0] , umacTensordims[1],umacTensordims[2] },options);
            umacTensor[1]=torch::zeros({1,umacTensordims[3] , umacTensordims[4],umacTensordims[5] },options);
            umacTensor[2]=torch::zeros({1,umacTensordims[6] , umacTensordims[7],umacTensordims[8] },options);
            Convert_StdArrMF_To_StdArrTensor(Unpack_umac(args...),umacTensor);
            Collectumac(args...,umacTensor);



            std::array<torch::Tensor,AMREX_SPACEDIM> RHSTensor;
            RHSTensor[0]=torch::zeros({1,srctermTensordim[0] , srctermTensordim[1],srctermTensordim[2] },options);
            RHSTensor[1]=torch::zeros({1,srctermTensordim[3] , srctermTensordim[4],srctermTensordim[5] },options);
            RHSTensor[2]=torch::zeros({1,srctermTensordim[6] , srctermTensordim[7],srctermTensordim[8] },options);



            std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
            TrimSourceMultiFab(args...,dmap,ba,source_termsTrimmed);
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSTensor); /* Convert Std::array<MultiFab,AMREX_SPACEDIM > to  std::array<torch::tensor, AMREX_SPACEDIM> */
            CollectRHS(args...,RHSTensor);


        }else if (RefineSol == true and TimeDataWindow[WindowIdx]>MovingAvg(TimeDataWindow) and ResidDataWindow[WindowIdx]>MovingAvg(ResidDataWindow) )
        {
            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);



            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensor;
            umacTensor[0]=torch::zeros({1,umacTensordims[0] , umacTensordims[1],umacTensordims[2] },options);
            umacTensor[1]=torch::zeros({1,umacTensordims[3] , umacTensordims[4],umacTensordims[5] },options);
            umacTensor[2]=torch::zeros({1,umacTensordims[6] , umacTensordims[7],umacTensordims[8] },options);
            Convert_StdArrMF_To_StdArrTensor(Unpack_umac(args...),umacTensor);
            Collectumac(args...,umacTensor);


            std::array<torch::Tensor,AMREX_SPACEDIM> RHSTensor;
            RHSTensor[0]=torch::zeros({1,srctermTensordim[0] , srctermTensordim[1],srctermTensordim[2] },options);
            RHSTensor[1]=torch::zeros({1,srctermTensordim[3] , srctermTensordim[4],srctermTensordim[5] },options);
            RHSTensor[2]=torch::zeros({1,srctermTensordim[6] , srctermTensordim[7],srctermTensordim[8] },options);

            std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
            TrimSourceMultiFab(args...,dmap,ba,source_termsTrimmed);
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSTensor); /* Convert Std::array<MultiFab,AMREX_SPACEDIM > to  std::array<torch::tensor, AMREX_SPACEDIM> */
            CollectRHS(args...,RHSTensor);
        }




        /* Train model */
        if(RefineSol == true)
        {
            torch::Tensor CheckNumSamples = Unpack_PresCollect(args...);

            /* Train model every "retrainFreq" number of steps during initial data collection period (size of moving average window) */
            if(step<(initNum+TimeDataWindow.size()) and step%retrainFreq==0)
            {
                TrainLoop(NETPres,Unpack_RHSCollect(args...),Unpack_PresCollect(args...),Unpack_umacCollect(args...),presTensordim,srctermTensordim,umacTensordims);

            /* Train model every time 3 new data points have been added to training set after initialization period */
            }else if ( (CheckNumSamples.size(0)-(initNum+TimeDataWindow.size()))%retrainFreq==0 )
            {
                TrainLoop(NETPres,Unpack_RHSCollect(args...),Unpack_PresCollect(args...),Unpack_umacCollect(args...),presTensordim,srctermTensordim,umacTensordims);
            }
        }

    };
    return new_function;
}








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
    MultiFab pres(ba, dmap, 1, 1);   /* ncomp=1, ngrow=1 */
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

    std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
        MultiFab::Copy(source_termsTrimmed[d], source_terms[d], 0, 0, 1, 1); 
    }



    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())  device = torch::Device(torch::kCUDA);


    /* Compute dimensions of single component pressure box */
    const auto & presbox  =   pres[0];
    IntVect presTensordim = presbox.bigEnd()-presbox.smallEnd()+1;



    /* Compute dimensions of each source term component box */
    std::vector<int> sourceTermTensordims(9);
    const auto & sourceTermXbox  =   source_termsTrimmed[0][0];
    const auto & sourceTermYbox  =   source_termsTrimmed[1][0];
    const auto & sourceTermZbox  =   source_termsTrimmed[2][0];
    IntVect srctermXTensordim = sourceTermXbox.bigEnd()-sourceTermXbox.smallEnd();
    IntVect srctermYTensordim = sourceTermYbox.bigEnd()-sourceTermYbox.smallEnd();
    IntVect srctermZTensordim = sourceTermZbox.bigEnd()-sourceTermZbox.smallEnd();
    for (int i=0; i<3;++i)
    {
        sourceTermTensordims[i  ]=srctermXTensordim[i]+1;
        sourceTermTensordims[i+3]=srctermYTensordim[i]+1;
        sourceTermTensordims[i+6]=srctermZTensordim[i]+1;
    }

    /* Compute dimensions of each umac component box */
    std::vector<int> umacTensordims(9);
    const auto & umacXbox  =   umac[0][0];
    const auto & umacYbox  =   umac[1][0];
    const auto & umacZbox  =   umac[2][0];
    IntVect umacXTensordim = umacXbox.bigEnd()-umacXbox.smallEnd();
    IntVect umacYTensordim = umacYbox.bigEnd()-umacYbox.smallEnd();
    IntVect umacZTensordim = umacZbox.bigEnd()-umacZbox.smallEnd();
    for (int i=0; i<3;++i)
    {
        umacTensordims[i  ]=umacXTensordim[i]+1;
        umacTensordims[i+3]=umacYTensordim[i]+1;
        umacTensordims[i+6]=umacZTensordim[i]+1;
    }

    int FlatdimIn= sourceTermTensordims[0]*sourceTermTensordims[1]*sourceTermTensordims[2] 
                    +sourceTermTensordims[3]*sourceTermTensordims[4]*sourceTermTensordims[5]
                    +sourceTermTensordims[6]*sourceTermTensordims[7]*sourceTermTensordims[8]
                    +presTensordim[0]*presTensordim[1]*presTensordim[2];

    int FlatdimOut= umacTensordims[0]*umacTensordims[1]*umacTensordims[2] 
                    +umacTensordims[3]*umacTensordims[4]*umacTensordims[5]
                    +umacTensordims[6]*umacTensordims[7]*umacTensordims[8]
                    +presTensordim[0]*presTensordim[1]*presTensordim[2];
    


    //   Define model and move to GPU
    auto TestNet = std::make_shared<Net>(FlatdimIn,FlatdimOut);
    TestNet->to(device);

    auto CNN_11= std::make_shared<StokesCNNet_11>(FlatdimIn,FlatdimOut);
    auto CNN_12= std::make_shared<StokesCNNet_12>(FlatdimIn,FlatdimOut);
    auto CNN_13= std::make_shared<StokesCNNet_13>(FlatdimIn,FlatdimOut);
    auto CNN_21= std::make_shared<StokesCNNet_21>(FlatdimIn,FlatdimOut);
    auto CNN_22= std::make_shared<StokesCNNet_22>(FlatdimIn,FlatdimOut);
    auto CNN_23= std::make_shared<StokesCNNet_23>(FlatdimIn,FlatdimOut);
    auto CNN_31= std::make_shared<StokesCNNet_31>(FlatdimIn,FlatdimOut);
    auto CNN_32= std::make_shared<StokesCNNet_32>(FlatdimIn,FlatdimOut);
    auto CNN_33= std::make_shared<StokesCNNet_33>(FlatdimIn,FlatdimOut);
    auto CNN_P1= std::make_shared<StokesCNNet_P1>(FlatdimIn,FlatdimOut);
    auto CNN_P2= std::make_shared<StokesCNNet_P2>(FlatdimIn,FlatdimOut);
    auto CNN_P3= std::make_shared<StokesCNNet_P3>(FlatdimIn,FlatdimOut);
    CNN_11->to(device);
    CNN_12->to(device);
    CNN_13->to(device);
    CNN_21->to(device);
    CNN_22->to(device);
    CNN_23->to(device);
    CNN_31->to(device);
    CNN_32->to(device);
    CNN_33->to(device);
    CNN_P1->to(device);
    CNN_P2->to(device);
    CNN_P3->to(device);


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
    auto advanceStokes_ML=Wrapper(advanceStokesPtr,RefineSol,device,TestNet,presTensordim,sourceTermTensordims,umacTensordims,dmap,ba) ;
    RefineSol=true;
    auto advanceStokes_ML2=Wrapper(advanceStokesPtr,RefineSol,device,TestNet,presTensordim,sourceTermTensordims,umacTensordims,dmap,ba) ;


    /* Initialize tensors that collect all pressure and source term data*/
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
    torch::Tensor presCollect= torch::zeros({1,presTensordim[0], presTensordim[1],presTensordim[2]},options);


    std::array<torch::Tensor,AMREX_SPACEDIM> RHSCollect;
    RHSCollect[0]=torch::zeros({1,sourceTermTensordims[0] , sourceTermTensordims[1],sourceTermTensordims[2] },options);
    RHSCollect[1]=torch::zeros({1,sourceTermTensordims[3] , sourceTermTensordims[4],sourceTermTensordims[5] },options);
    RHSCollect[2]=torch::zeros({1,sourceTermTensordims[6] , sourceTermTensordims[7],sourceTermTensordims[8] },options);

    std::array<torch::Tensor,AMREX_SPACEDIM> umacCollect;
    umacCollect[0]=torch::zeros({1,umacTensordims[0] , umacTensordims[1],umacTensordims[2] },options);
    umacCollect[1]=torch::zeros({1,umacTensordims[3] , umacTensordims[4],umacTensordims[5] },options);
    umacCollect[2]=torch::zeros({1,umacTensordims[6] , umacTensordims[7],umacTensordims[8] },options);


    std::vector<double> TimeDataWindow(50);
    std::vector<double> ResidDataWindow(50);


    /****************************************************************************
     *                                                                          *
     * Advance Time Steps                                                       *
     *                                                                          *
     ***************************************************************************/


    //___________________________________________________________________________

    MultiFab presDirect(ba, dmap, 1, 1); 
    pres.setVal(0.);  

    std::array< MultiFab, AMREX_SPACEDIM > umacDirect;
    defineFC(umacDirect, ba, dmap, 1);
    setVal(umacDirect, 0.);

    std::array<MultiFab, AMREX_SPACEDIM> source_termsDirect;
    for (int d=0; d<AMREX_SPACEDIM; ++d){
        source_termsDirect[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 6);
        source_termsDirect[d].setVal(0.);
    }



    int initNum     =20; //number of data points needed before making predictions. Note, must be set in wrapper as well.
    while (step < 3000)
    {
        // Spread forces to RHS
        std::array<MultiFab, AMREX_SPACEDIM> source_terms;
        for (int d=0; d<AMREX_SPACEDIM; ++d){
            source_terms[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 6);
            source_terms[d].setVal(0.);
        }


        RealVect f_0 = RealVect{dis(gen), dis(gen), dis(gen)};

        for (IBMarIter pti(ib_mc, ib_lev); pti.isValid(); ++pti) {

            // Get marker data (local to current thread)
            TileIndex index(pti.index(), pti.LocalTileIndex());
            AoS & markers = ib_mc.GetParticles(ib_lev).at(index).GetArrayOfStructs();
            long np = ib_mc.GetParticles(ib_lev).at(index).numParticles();

            for (int i =0; i<np; ++i) {
                ParticleType & mark = markers[i];
                mark.pos(0)=0.05*dis(gen);
                mark.pos(1)=0.05*dis(gen);
                mark.pos(2)=0*dis(gen);
                for (int d=0; d<AMREX_SPACEDIM; ++d)
                    mark.rdata(IBMReal::forcex + d) = f_0[d];
            }
        }
    

        // Spread to the `fc_force` multifab
        ib_mc.SpreadMarkers(0, source_terms);
        for (int d=0; d<AMREX_SPACEDIM; ++d)
            source_terms[d].SumBoundary(geom.periodicity());




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



        gmres::gmres_abs_tol = 1e-5;
        // Copy multifabs updated in stokes solver, then run stokes solver using
        // copied multifabs without ML wrapper. The time-to-solution is written to a text file. 
        // Quantities as "direct" correspond to this direct call of the Stokes solver
        MultiFab::Copy(presDirect, pres, 0, 0, 1, 1);
        for (int d=0; d<AMREX_SPACEDIM; ++d)
        {
            MultiFab::Copy(umacDirect[d], umac[d], 0, 0, 1, 1);
            MultiFab::Copy(source_termsDirect[d], source_terms[d], 0, 0, 1, 6);
        }
        // mimicing split in NN-wrapper (for comparison)
        if (step> initNum)
        {
            gmres_max_iter = 10 ;
            advanceStokes(
                    umacDirect, presDirect,              /* LHS */
                    mfluxdiv, source_termsDirect,  /* RHS */
                    alpha_fc, beta, gamma, beta_ed, geom, dt
                );
            gmres_max_iter = 100 ;
        }
        Real Direct_step_strt_time = ParallelDescriptor::second();
        advanceStokes(
                umacDirect, presDirect,              /* LHS */
                mfluxdiv, source_termsDirect,  /* RHS */
                alpha_fc, beta, gamma, beta_ed, geom, dt
            );
        Real  Direct_step_stop_time = ParallelDescriptor::second() -  Direct_step_strt_time;
        ParallelDescriptor::ReduceRealMax( Direct_step_stop_time);
        std::ofstream outfileDirect;
        outfileDirect.open("TimeDataDirect.txt", std::ios_base::app); // append instead of overwrite
        outfileDirect << Direct_step_stop_time << std::setw(10) << " \n"; 



        Real step_strt_time = ParallelDescriptor::second();

        // Print() << "COARSE SOLUTION" << "\n"; 
        advanceStokes_ML(umac,pres, /* LHS */
                        mfluxdiv,source_terms, /* RHS*/
                        alpha_fc, beta, gamma, beta_ed, geom, dt,
                        presCollect,RHSCollect,umacCollect,step,TimeDataWindow,ResidDataWindow /* ML */
                        );



                    /* Multifab check data */
                // int ncompPresTest=1;
                // int ngrowPrestTest=1;
                // // WriteTestMultiFab(0,time,geom,pres);
                // MultiFab mfdiff(pres.boxArray(), pres.DistributionMap(), ncompPresTest, ngrowPrestTest); 
                // MultiFab::Copy(mfdiff, pres, 0, 0, ncompPresTest, ngrowPrestTest); /* using same ncomp and ngrow as pres */
                // torch::Tensor presTensor = torch::zeros({presTensordim[0]+1, presTensordim[1]+1,presTensordim[2]+1},options);
                // ConvertToTensor(pres,presTensor);
                // TensorToMultifab(presTensor,pres);
                // // WriteTestMultiFab(1,time,geom,pres);
                // MultiFab::Subtract(mfdiff, pres, 0, 0, ncompPresTest, ngrowPrestTest);
                // for (int icomp = 0; icomp < ncompPresTest; ++icomp) {
                //     Print() << "Component " << icomp << std::endl; 
                //     Print() << "diff Min,max: " << mfdiff.min(icomp,ngrowPrestTest) 
                //     << " , " << mfdiff.max(icomp,ngrowPrestTest) << std::endl;
                // }




        // Print() << "REFINE SOLUTION" << "\n"; 
        
        gmres::gmres_abs_tol = 1e-6;
        advanceStokes_ML2(umac,pres, /* LHS */
                        mfluxdiv,source_terms,/* RHS */
                        alpha_fc, beta, gamma, beta_ed, geom, dt,
                        presCollect,RHSCollect,umacCollect,step,TimeDataWindow,ResidDataWindow /* ML */
                        );

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        time = time + dt;
        step ++;
        // write out umac & pres to a plotfile
        WritePlotFile(step, time, geom, umac, pres, ib_mc);

        if(step >9)
        {
            CNN_TrainLoop(CNN_11,CNN_12,CNN_13,CNN_21,CNN_22,CNN_23,
                            CNN_31,CNN_32,CNN_33,CNN_P1,CNN_P2,CNN_P3,
                            RHSCollect,presCollect,umacCollect,presTensordim,
                            sourceTermTensordims,umacTensordims);
        }

    }




    // // Call the timer again and compute the maximum difference between the start
    // // time and stop time over all processors
    // Real stop_time = ParallelDescriptor::second() - strt_time;
    // ParallelDescriptor::ReduceRealMax(stop_time);
    // amrex::Print() << "Run time = " << stop_time << std::endl;
}
