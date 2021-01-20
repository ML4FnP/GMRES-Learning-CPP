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
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* Stokes CNN */
struct StokesCNNet_Ux : torch::nn::Module {
  StokesCNNet_Ux( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf12(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf13(torch::nn::Conv3dOptions(1 , 1 , 5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf17(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf18(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf19(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       LRELU(torch::nn::LeakyReLUOptions().negative_slope(0.01)),
       BNorm3D1(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D2(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D3(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D4(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D5(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);
        register_module("convf17", convf17);
        register_module("convf18", convf18);
        register_module("convf19", convf19);
        register_module("LRELU", LRELU);
        register_module("BNorm3D1", BNorm3D1);
        register_module("BNorm3D2", BNorm3D2);
        register_module("BNorm3D3", BNorm3D3);
        register_module("BNorm3D4", BNorm3D4);
        register_module("BNorm3D5", BNorm3D5);
    }

   torch::Tensor forward(torch::Tensor Phi)
   {
        int64_t Current_batchsize= Phi.size(0);
        Phi = LRELU(BNorm3D1(convf11(Phi.unsqueeze(1))));
        Phi = LRELU(BNorm3D2(convf12(Phi)));
        Phi = LRELU(BNorm3D3(convf13(Phi)));
        Phi = LRELU(BNorm3D4(convf14(Phi)));
        Phi = LRELU((convf15(Phi)));
        Phi = LRELU((convf16(Phi)));
        Phi = LRELU((convf17(Phi)));
        Phi = LRELU((convf18(Phi)));
        Phi = ((convf19(Phi)));
        torch::Tensor UX = Phi.squeeze(1);
        return UX;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16,convf17,convf18,convf19;
   torch::nn::LeakyReLU LRELU;
   torch::nn::BatchNorm3d BNorm3D1,BNorm3D2,BNorm3D3,BNorm3D4,BNorm3D5;

};
//////////////////////////////////////////////////////////////////////////////
struct StokesCNNet_Uy : torch::nn::Module {
  StokesCNNet_Uy( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf12(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf13(torch::nn::Conv3dOptions(1 , 1 , 5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf17(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf18(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf19(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       LRELU(torch::nn::LeakyReLUOptions().negative_slope(0.01)),
       BNorm3D1(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D2(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D3(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D4(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D5(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);
        register_module("convf17", convf17);
        register_module("convf18", convf18);
        register_module("convf19", convf19);
        register_module("LRELU", LRELU);
        register_module("BNorm3D1", BNorm3D1);
        register_module("BNorm3D2", BNorm3D2);
        register_module("BNorm3D3", BNorm3D3);
        register_module("BNorm3D4", BNorm3D4);
        register_module("BNorm3D5", BNorm3D5);
    }

   torch::Tensor forward(torch::Tensor Phi)
   {
        int64_t Current_batchsize= Phi.size(0);
        Phi = LRELU(BNorm3D1(convf11(Phi.unsqueeze(1))));
        Phi = LRELU(BNorm3D2(convf12(Phi)));
        Phi = LRELU(BNorm3D3(convf13(Phi)));
        Phi = LRELU(BNorm3D4(convf14(Phi)));
        Phi = LRELU((convf15(Phi)));
        Phi = LRELU((convf16(Phi)));
        Phi = LRELU((convf17(Phi)));
        Phi = LRELU((convf18(Phi)));
        Phi = ((convf19(Phi)));
        torch::Tensor UY = Phi.squeeze(1);
        return UY;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16,convf17,convf18,convf19;
   torch::nn::LeakyReLU LRELU;
   torch::nn::BatchNorm3d BNorm3D1,BNorm3D2,BNorm3D3,BNorm3D4,BNorm3D5;

};
//////////////////////////////////////////////////////////////////////////////
struct StokesCNNet_Uz : torch::nn::Module {
  StokesCNNet_Uz( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf12(torch::nn::Conv3dOptions(1 , 1,  5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf13(torch::nn::Conv3dOptions(1 , 1 , 5).stride(1).padding(6).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf17(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf18(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf19(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       LRELU(torch::nn::LeakyReLUOptions().negative_slope(0.01)),
       BNorm3D1(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D2(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D3(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D4(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true)),
       BNorm3D5(torch::nn::BatchNorm3dOptions(1).affine(true).track_running_stats(true))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);
        register_module("convf17", convf17);
        register_module("convf18", convf18);
        register_module("convf19", convf19);
        register_module("LRELU", LRELU);
        register_module("BNorm3D1", BNorm3D1);
        register_module("BNorm3D2", BNorm3D2);
        register_module("BNorm3D3", BNorm3D3);
        register_module("BNorm3D4", BNorm3D4);
        register_module("BNorm3D5", BNorm3D5);
    }

   torch::Tensor forward(torch::Tensor Phi)
   {
        int64_t Current_batchsize= Phi.size(0);
        Phi = LRELU(BNorm3D1(convf11(Phi.unsqueeze(1))));
        Phi = LRELU(BNorm3D2(convf12(Phi)));
        Phi = LRELU(BNorm3D3(convf13(Phi)));
        Phi = LRELU(BNorm3D4(convf14(Phi)));
        Phi = LRELU((convf15(Phi)));
        Phi = LRELU((convf16(Phi)));
        Phi = LRELU((convf17(Phi)));
        Phi = LRELU((convf18(Phi)));
        Phi = ((convf19(Phi)));
        torch::Tensor UZ = Phi.squeeze(1);
        return UZ;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16,convf17,convf18,convf19;
   torch::nn::LeakyReLU LRELU;
   torch::nn::BatchNorm3d BNorm3D1,BNorm3D2,BNorm3D3,BNorm3D4,BNorm3D5;

};

////////////////////////////////////////////////////////////////////////////////
/* Pressure CNN */
struct StokesCNNet_P : torch::nn::Module {
  StokesCNNet_P( )
    :  convf11(torch::nn::Conv3dOptions(1 , 8,  5).stride(1).padding(4).dilation(2).groups(1).bias(false).padding_mode(torch::kReplicate)),
       convf12(torch::nn::Conv3dOptions(8 , 8,  5).stride(1).padding(4).dilation(2).groups(1).bias(false).padding_mode(torch::kReplicate)),
       convf13(torch::nn::Conv3dOptions(8 , 4 , 5).stride(1).padding(4).dilation(2).groups(1).bias(false).padding_mode(torch::kReplicate)),
       convf14(torch::nn::Conv3dOptions(4 , 4 , 5).stride(1).padding(4).dilation(2).groups(1).bias(false).padding_mode(torch::kReplicate)),
       convf15(torch::nn::Conv3dOptions(4 , 2 , 5).stride(1).padding(2).dilation(1).groups(1).bias(false).padding_mode(torch::kReplicate)),
       convf16(torch::nn::Conv3dOptions(2 , 1 , 5).stride(1).padding(2).dilation(1).groups(1).bias(false).padding_mode(torch::kReplicate)),
    //    convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf17(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf18(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf19(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf110(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf111(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf112(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf113(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf114(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf115(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf116(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf117(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf118(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf119(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
    //    convf120(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kReplicate)),
       convf121(torch::nn::Conv3dOptions(1 , 1 , 5).stride(1).padding(2).dilation(1).groups(1).bias(false).padding_mode(torch::kReplicate)),


       LRELU(torch::nn::LeakyReLUOptions().negative_slope(1)),
       BNorm3D1(torch::nn::BatchNorm3dOptions(8).affine(true).track_running_stats(true)),
       BNorm3D2(torch::nn::BatchNorm3dOptions(8).affine(true).track_running_stats(true)),
       BNorm3D3(torch::nn::BatchNorm3dOptions(4).affine(true).track_running_stats(true)),
       BNorm3D4(torch::nn::BatchNorm3dOptions(4).affine(true).track_running_stats(true)),
       BNorm3D5(torch::nn::BatchNorm3dOptions(2).affine(true).track_running_stats(true))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);
        // register_module("convf17", convf17);
        // register_module("convf18", convf18);
        // register_module("convf19", convf19);
        // register_module("convf110", convf110);
        // register_module("convf111", convf111);
        // register_module("convf112", convf112);
        // register_module("convf113", convf113);
        // register_module("convf114", convf114);
        // register_module("convf115", convf115);
        // register_module("convf116", convf116);
        // register_module("convf117", convf117);
        // register_module("convf118", convf118);
        // register_module("convf119", convf119);
        // register_module("convf120", convf120);
        register_module("convf121", convf121);
        register_module("LRELU", LRELU);
        register_module("BNorm3D1", BNorm3D1);
        register_module("BNorm3D2", BNorm3D2);
        register_module("BNorm3D3", BNorm3D3);
        register_module("BNorm3D4", BNorm3D4);
        register_module("BNorm3D5", BNorm3D5);
    }

   torch::Tensor forward(torch::Tensor divF)
   {
        int64_t Current_batchsize= divF.size(0);
        divF = LRELU(BNorm3D1(convf11(divF.unsqueeze(1))));
        divF = LRELU(BNorm3D2(convf12(divF)));
        divF = LRELU(BNorm3D3(convf13(divF)));
        divF = LRELU(BNorm3D4(convf14(divF)));
        divF = LRELU(BNorm3D5(convf15(divF)));
        divF = LRELU((convf16(divF)));
        // divF = LRELU((convf17(divF)));
        // divF = LRELU((convf18(divF)));
        // divF = LRELU((convf19(divF)));
        // divF = LRELU((convf110(divF)));
        // divF = LRELU((convf111(divF)));
        // divF = LRELU((convf112(divF)));
        // divF = LRELU((convf113(divF)));
        // divF = LRELU((convf114(divF)));
        // divF = LRELU((convf115(divF)));
        // divF = LRELU((convf116(divF)));
        // divF = LRELU((convf117(divF)));
        // divF = LRELU((convf118(divF)));
        // divF = LRELU((convf119(divF)));
        // divF = LRELU((convf120(divF)));
        divF = ((convf121(divF)));

        torch::Tensor P = divF.squeeze(1);
        return P;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16;
   //,convf16,convf17,convf18,convf19;
   // convf117,convf118,convf119,convf120,
   // torch::nn::Conv3d convf110,convf111,convf112,convf113,convf114,convf115,convf116;
   torch::nn::Conv3d convf121;
   torch::nn::LeakyReLU LRELU;
   torch::nn::BatchNorm3d BNorm3D1,BNorm3D2,BNorm3D3,BNorm3D4,BNorm3D5;

};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* Dataset class for pressure CNN */
class CustomDatasetCNN_Pres : public torch::data::Dataset<CustomDatasetCNN_Pres>
{
    private:
        torch::Tensor bTensor, SolTensor;
    public:
        CustomDatasetCNN_Pres( torch::Tensor DivSrcTensor, torch::Tensor Pres,
                            const IntVect DivFdim,  const IntVect presTensordim)
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
            int64_t Current_batchsize= DivSrcTensor[0].size(0);

            /* Add channel dim to ever tensor component of the Std::array<torch::tensor,Amrex_dim> objects */
            DivSrcTensor=DivSrcTensor;
            Pres=Pres;
            /* Network input tensor */
            bTensor=DivSrcTensor;
            /* Network solution tensor */
            SolTensor=Pres;    
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

  //////////////////////////////////////////////////////////////////////////////
/* Dataset class for scalar Ux CNN */
class CustomDatasetCNN_Ux : public torch::data::Dataset<CustomDatasetCNN_Ux>
{
    private:
        torch::Tensor bTensor, SolTensor;
    public:
        CustomDatasetCNN_Ux(torch::Tensor SrcTensor,torch::Tensor pgTensor, torch::Tensor UX)
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
            int64_t Current_batchsize= SrcTensor[0].size(0);

            /* Network input tensor */
            bTensor=SrcTensor-pgTensor;
            /* Network solution tensor */
            SolTensor=UX;    
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
    //////////////////////////////////////////////////////////////////////////////
/* Dataset class for scalar Uy CNN */
class CustomDatasetCNN_Uy : public torch::data::Dataset<CustomDatasetCNN_Uy>
{
    private:
        torch::Tensor bTensor, SolTensor;
    public:
        CustomDatasetCNN_Uy(torch::Tensor SrcTensor,torch::Tensor pgTensor, torch::Tensor UY)
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
            int64_t Current_batchsize= SrcTensor[0].size(0);

            /* Network input tensor */
            bTensor=SrcTensor-pgTensor;
            /* Network solution tensor */
            SolTensor=UY;    
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
//////////////////////////////////////////////////////////////////////////////
  /* Dataset class for scalar Uz CNN */
class CustomDatasetCNN_Uz : public torch::data::Dataset<CustomDatasetCNN_Uz>
{
    private:
        torch::Tensor bTensor, SolTensor;
    public:
        CustomDatasetCNN_Uz(torch::Tensor SrcTensor,torch::Tensor pgTensor, torch::Tensor UZ)
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
            int64_t Current_batchsize= SrcTensor[0].size(0);

            /* Network input tensor */
            bTensor=SrcTensor-pgTensor;
            /* Network solution tensor */
            SolTensor=UZ;    
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* Multifab/Tensor interfaces */

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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
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


    Real norm_b;            // |b|;           
    Real norm_pre_b;        // |M^-1 b|;      
    Real norm_resid_Stokes; // |b-Ax|;        
    Real norm_u_noprecon;   // u component of norm_resid_Stokes
    Real norm_p_noprecon;   // p component of norm_resid_Stokes

    Real norm_u; // temporary norms used to build full-state norm
    Real norm_p; // temporary norms used to build full-state norm


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

        // gmres_rhs_u,gmres_rhs_p,umac,pres,
        // bu,bp,xu,xp

}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* Functions for unpacking parameter pack passed to wrapped function. */

std::array< MultiFab, AMREX_SPACEDIM >& Unpack_umac(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  umacCollect;
}

std::array<torch::Tensor,AMREX_SPACEDIM>& Unpack_pgCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  pgCollect;
}


torch::Tensor& Unpack_DivFCollect(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  DivFCollect;
}


int unpack_Step(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  

       return TimeDataWindow;
    
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* functions that require something from the parameter pack ( or act on the pack) but do not simply unpack values are below */

void CollectPressure(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::array<torch::Tensor,AMREX_SPACEDIM> umacFinal)
{  
        for (int d=0; d<AMREX_SPACEDIM; ++d)
        {
            umacCollect[d]=torch::cat({umacCollect[d],umacFinal[d]},0);
        }
}

void CollectDivF(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,torch::Tensor DivFfinal)
{  

        DivFCollect=torch::cat({DivFCollect,DivFfinal},0);
    
}

void Collectpg(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,std::array<torch::Tensor,AMREX_SPACEDIM> pgFinal)
{  
        for (int d=0; d<AMREX_SPACEDIM; ++d)
        {
            pgCollect[d]=torch::cat({pgCollect[d],pgFinal[d]},0);
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect,
                   std::array<torch::Tensor,AMREX_SPACEDIM>&  pgCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,amrex::DistributionMapping dmap, BoxArray  ba, std::array<MultiFab, AMREX_SPACEDIM>& source_termsTrimmed)
{  
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
    }
    setVal(source_termsTrimmed, 0.);
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        MultiFab::Copy(source_termsTrimmed[d], sourceTerms[d], 0, 0, 1, 1); 
    }
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void  CNN_P_TrainLoop( std::shared_ptr<StokesCNNet_P> CNN_P,
                       torch::Tensor& DivSrcTensorCollect,torch::Tensor& PresCollect,
                       const IntVect presTensordim, const IntVect DivFdim)
{
                /*Setting up learning loop below */
                // torch::optim::SGD optimizerPres({CNN_P->parameters()}, torch::optim::SGDOptions (0.01));
                // torch::optim::AdamW optimizerPres({CNN_P->parameters()});
                torch::optim::Adagrad optimizerPres({CNN_P->parameters()}, torch::optim::AdagradOptions (0.01));

                

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN_Pres(DivSrcTensorCollect,PresCollect,DivFdim,presTensordim).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 16;
                float e1 = 1e-13;
                int64_t epoch = 0;
                int64_t numEpoch = 500; 

                float lossP  = 10.0;
                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while((lossP)>e1 and epoch<numEpoch )
                {


                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 

                            // Solution data
                            torch::Tensor target = batch.target; 
   
                            // Reset gradients
                            optimizerPres.zero_grad();

                            // forward pass
                            torch::Tensor outputP = CNN_P->forward(data.to(torch::kFloat32));

                            //evaulate loss                        
                            // auto loss_out = torch::nn::functional::mse_loss(output,target, torch::nn::functional::MSELossFuncOptions(torch::kSum));
                            torch::Tensor loss_outP = torch::mse_loss(outputP,target.to(torch::kFloat32));  

                            // Extract loss value for printing to console
                            lossP  = loss_outP.item<float>();

                            // Backward pass
                            loss_outP.backward();


                            // Apply gradients
                            optimizerPres.step();
                            // Print loop info to console
                    }
                    epoch = epoch +1;
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss P : "  << lossP << std::endl;
                    std::cout << " Epochs 5: " << epoch << std::endl;
                }

}



void  CNN_UX_TrainLoop( std::shared_ptr<StokesCNNet_Ux> CNN_UX,
                       torch::Tensor& RHSCollect,torch::Tensor& pgCollect,
                       torch::Tensor& umacCollect)
{
                /*Setting up learning loop below */
                // torch::optim::SGD optimizerUx({CNN_UX->parameters()}, torch::optim::SGDOptions (0.01));
                torch::optim::AdamW optimizerUx({CNN_UX->parameters()});


                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN_Ux(RHSCollect,pgCollect,umacCollect).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 1000; 

                float lossUx  = 10.0;
                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while(lossUx>e1 and epoch<numEpoch )
                {


                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 

                            // Solution data
                            torch::Tensor target = batch.target; 
   
                            // Reset gradients
                            optimizerUx.zero_grad();

                            // forward pass
                            torch::Tensor outputUx = CNN_UX->forward(data.to(torch::kFloat32));

                            //evaulate loss                        
                            // torch::Tensor loss_outUx = torch::mse_loss(outputUx,target.to(torch::kFloat32));  

                            torch::Tensor loss_outUx =  torch::nn::functional::mse_loss(outputUx,target.to(torch::kFloat32));

                            // Extract loss value for printing to console
                            lossUx  = loss_outUx.item<float>();

                            // Backward pass
                            loss_outUx.backward();


                            // Apply gradients
                            optimizerUx.step();

                            epoch = epoch +1;

                            // Print loop info to console
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss Ux : "  << lossUx << std::endl;
                    std::cout << "Batch Number: " << epoch << std::endl;
                }

}

void  CNN_UY_TrainLoop( std::shared_ptr<StokesCNNet_Uy> CNN_UY,
                       torch::Tensor& RHSCollect,torch::Tensor& pgCollect,
                       torch::Tensor& umacCollect)
{
                /*Setting up learning loop below */
                torch::optim::AdamW optimizerUy({CNN_UY->parameters()});

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN_Uy(RHSCollect,pgCollect,umacCollect).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 1000; 

                float lossUy  = 10.0;
                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while(lossUy>e1 and epoch<numEpoch )
                {


                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 

                            // Solution data
                            torch::Tensor target = batch.target; 
   
                            // Reset gradients
                            optimizerUy.zero_grad();

                            // forward pass
                            torch::Tensor outputUy = CNN_UY->forward(data.to(torch::kFloat32));

                            //evaulate loss                        
                            torch::Tensor loss_outUy = torch::mse_loss(outputUy,target.to(torch::kFloat32));  

                            // Extract loss value for printing to console
                            lossUy  = loss_outUy.item<float>();

                            // Backward pass
                            loss_outUy.backward();

                            // Apply gradients
                            optimizerUy.step();

                            epoch = epoch +1;

                            // Print loop info to console
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss Uy : "  << lossUy << std::endl;
                    std::cout << "Batch Number: " << epoch << std::endl;
                }

}



void  CNN_UZ_TrainLoop( std::shared_ptr<StokesCNNet_Uz> CNN_UZ,
                       torch::Tensor& RHSCollect,torch::Tensor& pgCollect,
                       torch::Tensor& umacCollect)
{
                /*Setting up learning loop below */
                torch::optim::AdamW optimizerUz({CNN_UZ->parameters()});

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN_Uz(RHSCollect,pgCollect,umacCollect).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 1000; 

                float lossUz  = 10.0;
                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while(lossUz>e1 and epoch<numEpoch )
                {


                    for(torch::data::Example<>& batch: *data_loader) 
                    {      // RHS data
                            torch::Tensor data = batch.data; 

                            // Solution data
                            torch::Tensor target = batch.target; 
   
                            // Reset gradients
                            optimizerUz.zero_grad();

                            // forward pass
                            torch::Tensor outputUz = CNN_UZ->forward(data.to(torch::kFloat32));

                            //evaulate loss                        
                            torch::Tensor loss_outUz = torch::mse_loss(outputUz,target.to(torch::kFloat32));  

                            // Extract loss value for printing to console
                            lossUz  = loss_outUz.item<float>();

                            // Backward pass
                            loss_outUz.backward();

                            // Apply gradients
                            optimizerUz.step();

                            epoch = epoch +1;

                            // Print loop info to console
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss UZ : "  << lossUz << std::endl;
                    std::cout << "Batch Number: " << epoch << std::endl;
                }

}
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


/* ML Wrapper for advanceStokes */
template<typename F>
auto Wrapper(F func,bool RefineSol ,torch::Device device,
                std::shared_ptr<StokesCNNet_Ux> CNN_UX,std::shared_ptr<StokesCNNet_Uy> CNN_UY,
                std::shared_ptr<StokesCNNet_Uz> CNN_UZ,std::shared_ptr<StokesCNNet_P> CNN_P,                 
                const IntVect presTensordim, const std::vector<int> srctermTensordim, 
                const std::vector<int> umacTensordims,const IntVect DivFdim,
                const std::vector<int> pgTensordims,
                amrex::DistributionMapping dmap, BoxArray  ba)
{
    auto new_function = [func,RefineSol,device,CNN_UX,CNN_UY,CNN_UZ,CNN_P,presTensordim,srctermTensordim,umacTensordims,DivFdim,pgTensordims,dmap,ba](auto&&... args)
    {
        int retrainFreq =32;
        int initNum     =64;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
        int step=unpack_Step(args...);
        std::vector<double> TimeDataWindow =unpack_TimeDataWindow(args...);
        std::vector<double> ResidDataWindow =unpack_ResidDataWindow(args...);
        int WindowIdx=((step-initNum)-1)%TimeDataWindow.size();
        bool use_NN_prediction =false;


        MultiFab presNN(ba, dmap, 1, 1); 
        presNN.setVal(0.);  
        // MultiFabPhysBC(presNN, geom, 0, 1, 0);     // varType  0: pressure+

        std::array< MultiFab, AMREX_SPACEDIM > umacNN;
        defineFC(umacNN, ba, dmap, 1);
        setVal(umacNN, 0.);


        /*Compute DivF with ghost cells consistent with pressure (ghost cells set to 0)*/
        MultiFab DivF(ba, dmap, 1, 1);   
        DivF.setVal(0);
        ComputeDiv(DivF,Unpack_sourceTerms(args...),0,0,1,Unpack_geom(args...),0);


        /* Use NN to predict pressure */
        if (RefineSol==false and step>initNum )
        {

            std::array<torch::Tensor,AMREX_SPACEDIM> RHSTensor;
            RHSTensor[0]=torch::zeros({1,srctermTensordim[0] , srctermTensordim[1],srctermTensordim[2] },options);
            RHSTensor[1]=torch::zeros({1,srctermTensordim[3] , srctermTensordim[4],srctermTensordim[5] },options);
            RHSTensor[2]=torch::zeros({1,srctermTensordim[6] , srctermTensordim[7],srctermTensordim[8] },options);
            
            std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
            TrimSourceMultiFab(args...,dmap,ba,source_termsTrimmed);
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSTensor); /* Convert Std::array<MultiFab,AMREX_SPACEDIM > to  std::array<torch::tensor, AMREX_SPACEDIM> *


            /* Convert divF into a single tensor */
            torch::Tensor DivFTensor= torch::zeros({1,DivFdim[0],DivFdim[1],DivFdim[2] },options);
            ConvertToTensor(DivF,DivFTensor);

            /* Get Pressure prediction as tensor */
            torch::Tensor presTensor = CNN_P->forward(DivFTensor.to(torch::kFloat32));

            /* Convert pressure tensor to multifab using distribution map of original pressure MultiFab */
            TensorToMultifab(presTensor,presNN);

            /* Compute pressure gradient from NN output after being converted to a multifab */
            std::array< MultiFab, AMREX_SPACEDIM > pg;
            for (int d=0; d<AMREX_SPACEDIM; d++)
            {
                pg[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
            }
            for (int d=0; d<AMREX_SPACEDIM; ++d)
            {
                pg[d].setVal(0);
            } 
            SetPressureBC(presNN, Unpack_geom(args...));
            ComputeGrad(presNN, pg, 0, 0, 1, 0, Unpack_geom(args...));

            /* Convert pressure gradient MultiFab to PyTorch Tensor */
            std::array< torch::Tensor, AMREX_SPACEDIM > pgNNTensor;
            pgNNTensor[0]=torch::zeros({1,pgTensordims[0] , pgTensordims[1],pgTensordims[2] },options);
            pgNNTensor[1]=torch::zeros({1,pgTensordims[3] , pgTensordims[4],pgTensordims[5] },options);
            pgNNTensor[2]=torch::zeros({1,pgTensordims[6] , pgTensordims[7],pgTensordims[8] },options);
            Convert_StdArrMF_To_StdArrTensor(pg,pgNNTensor);

            /* Get prediction as tensor */
            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensor;
            umacTensor[0] = CNN_UX->forward((RHSTensor[0]-pgNNTensor[0])).to(torch::kFloat32);
            umacTensor[1] = CNN_UY->forward((RHSTensor[1]-pgNNTensor[1])).to(torch::kFloat32);                              
            umacTensor[2] = CNN_UZ->forward((RHSTensor[2]-pgNNTensor[2])).to(torch::kFloat32);

            /* Convert std::array tensor to std::array multifab using distribution map of original MultiFab */
            stdArrTensorTostdArrMultifab(umacTensor,umacNN);


            /* Set up copies for intermediate direct calculation */
            std::array< MultiFab, AMREX_SPACEDIM > umacDirect;
            defineFC(umacDirect, ba, dmap, 1);
            setVal(umacDirect, 0.);
            for (int d=0; d<AMREX_SPACEDIM; ++d)
            {
                MultiFab::Copy(umacDirect[d], Unpack_umac(args...)[d], 0, 0, 1, 1);
            }
            MultiFab presDirect(ba, dmap, 1, 1); 
            presDirect.setVal(0.); 
            MultiFab::Copy(presDirect,Unpack_pres(args...), 0, 0, 1, 1);


            /* Set up copies for intermediate ML-assisted calculation */
            std::array< MultiFab, AMREX_SPACEDIM > umacML;
            defineFC(umacML, ba, dmap, 1);
            setVal(umacML, 0.);
            for (int d=0; d<AMREX_SPACEDIM; ++d)
            {
                MultiFab::Copy(umacML[d], umacNN[d], 0, 0, 1, 1);
            }
            MultiFab presML(ba, dmap, 1, 1); 
            presML.setVal(0.); 
            MultiFab::Copy(presML,presNN, 0, 0, 1, 1);


            Real norm_residNN;
            Real norm_resid;

            // gmres_max_iter=5;

            // func(umacML,presML,Unpack_flux(args...),Unpack_sourceTerms(args...),
            //     Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
            //     Unpack_geom(args...),Unpack_dt(args...));

            ResidCompute(umacML,presML,Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...),norm_residNN);

            // func(umacDirect,presDirect,Unpack_flux(args...),Unpack_sourceTerms(args...),
            //     Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
            //     Unpack_geom(args...),Unpack_dt(args...));

            ResidCompute(umacDirect,presDirect,Unpack_flux(args...),Unpack_sourceTerms(args...),
                Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
                Unpack_geom(args...),Unpack_dt(args...),norm_resid);

            // gmres_max_iter=100;

            amrex::Print() <<  "Direct resid "<<  norm_resid << " *****************" << " \n";
            amrex::Print() <<  "NN resid "<<  norm_residNN << " *****************" << " \n";

            if (norm_residNN < norm_resid)
            {
                amrex::Print() << "Use guess provided by NN" << " \n";
                use_NN_prediction=true;
                for (int d=0; d<AMREX_SPACEDIM; ++d)
                {
                    MultiFab::Copy(Unpack_umac(args...)[d],umacML[d],0, 0, 1, 1);
                }
                MultiFab::Copy(Unpack_pres(args...),presML, 0, 0, 1, 1);
            }else if(norm_residNN > norm_resid)
            {
                amrex::Print() << "1st GMRES resid for NN guess is too high. Discarding NN guess " << " \n";
                use_NN_prediction=false;
                for (int d=0; d<AMREX_SPACEDIM; ++d)
                {
                    MultiFab::Copy(Unpack_umac(args...)[d],umacDirect[d],0, 0, 1, 1);
                } 
                MultiFab::Copy(Unpack_pres(args...),presDirect, 0, 0, 1, 1);
            }
            ResidDataWindow[WindowIdx]= norm_residNN; /* Add residual to window of values */
        }

        /* Evaluate wrapped function with either the NN prediction or original input */
        if(RefineSol==false and step>initNum)
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


            /* Create  pressure gradient MultiFab */
            std::array< MultiFab, AMREX_SPACEDIM > pg;
            for (int d=0; d<AMREX_SPACEDIM; d++)
            {
                pg[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
            }
            for (int d=0; d<AMREX_SPACEDIM; ++d)
            {
                pg[d].setVal(0);
            }
            /* Compute  pressure gradient MultiFab */
            ComputeGrad(Unpack_pres(args...), pg, 0, 0, 1, 0,Unpack_geom(args...));



            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);


            std::array<torch::Tensor,AMREX_SPACEDIM> pgTensor;
            pgTensor[0]=torch::zeros({1,pgTensordims[0] , pgTensordims[1],pgTensordims[2] },options);
            pgTensor[1]=torch::zeros({1,pgTensordims[3] , pgTensordims[4],pgTensordims[5] },options);
            pgTensor[2]=torch::zeros({1,pgTensordims[6] , pgTensordims[7],pgTensordims[8] },options);
            Convert_StdArrMF_To_StdArrTensor(pg,pgTensor);
            Collectpg(args...,pgTensor);


            torch::Tensor DivFTensor= torch::zeros({1,DivFdim[0],DivFdim[1],DivFdim[2] },options);
            ConvertToTensor(DivF,DivFTensor);
            CollectDivF(args...,DivFTensor);


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

            /* Create  pressure gradient MultiFab */
            std::array< MultiFab, AMREX_SPACEDIM > pg;
            for (int d=0; d<AMREX_SPACEDIM; d++)
            {
                pg[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
            }
            for (int d=0; d<AMREX_SPACEDIM; ++d)
            {
                pg[d].setVal(0);
            }
            /* Compute  pressure gradient MultiFab */
            ComputeGrad(Unpack_pres(args...), pg, 0, 0, 1, 0,Unpack_geom(args...));


            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);

            std::array<torch::Tensor,AMREX_SPACEDIM> pgTensor;
            pgTensor[0]=torch::zeros({1,pgTensordims[0] , pgTensordims[1],pgTensordims[2] },options);
            pgTensor[1]=torch::zeros({1,pgTensordims[3] , pgTensordims[4],pgTensordims[5] },options);
            pgTensor[2]=torch::zeros({1,pgTensordims[6] , pgTensordims[7],pgTensordims[8] },options);
            Convert_StdArrMF_To_StdArrTensor(pg,pgTensor);
            Collectpg(args...,pgTensor);

            torch::Tensor DivFTensor= torch::zeros({1,DivFdim[0],DivFdim[1],DivFdim[2] },options);
            ConvertToTensor(DivF,DivFTensor);
            CollectDivF(args...,DivFTensor);

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
            int SampleIndex=(CheckNumSamples.size(0)-(initNum+TimeDataWindow.size()));


            /* Train model every "retrainFreq" number of steps during initial data collection period (size of moving average window) */
            if(step<(initNum+TimeDataWindow.size()) and step%retrainFreq==0 and step>0)
            {
                CNN_P_TrainLoop(CNN_P, Unpack_DivFCollect(args...), Unpack_PresCollect(args...),presTensordim, DivFdim);
                CNN_UX_TrainLoop(CNN_UX,Unpack_RHSCollect(args...)[0],Unpack_pgCollect(args...)[0],Unpack_umacCollect(args...)[0]);
                CNN_UY_TrainLoop(CNN_UY,Unpack_RHSCollect(args...)[1],Unpack_pgCollect(args...)[1],Unpack_umacCollect(args...)[1]);
                CNN_UZ_TrainLoop(CNN_UZ,Unpack_RHSCollect(args...)[2],Unpack_pgCollect(args...)[2],Unpack_umacCollect(args...)[2]);

            /* Train model every time 3 new data points have been added to training set after initialization period */
            }else if ( SampleIndex>0 and SampleIndex%retrainFreq==0 )
            {
                CNN_P_TrainLoop(CNN_P, Unpack_DivFCollect(args...), Unpack_PresCollect(args...),presTensordim, DivFdim);
                CNN_UX_TrainLoop(CNN_UX,Unpack_RHSCollect(args...)[0],Unpack_pgCollect(args...)[0],Unpack_umacCollect(args...)[0]);
                CNN_UY_TrainLoop(CNN_UY,Unpack_RHSCollect(args...)[1],Unpack_pgCollect(args...)[1],Unpack_umacCollect(args...)[1]);
                CNN_UZ_TrainLoop(CNN_UZ,Unpack_RHSCollect(args...)[2],Unpack_pgCollect(args...)[2],Unpack_umacCollect(args...)[2]);
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
            mark.pos(0)=0.85*dis(gen);
            mark.pos(1)=0.85*dis(gen);
            mark.pos(2)=0.85*dis(gen);
            for (int d=0; d<AMREX_SPACEDIM; ++d)
                mark.rdata(IBMReal::forcex + d) = f_0[d];
        }
    }
    
    ib_mc.SpreadMarkers(0, source_terms);
    for (int d=0; d<AMREX_SPACEDIM; ++d)
        source_terms[d].SumBoundary(geom.periodicity());


    std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
        MultiFab::Copy(source_termsTrimmed[d], source_terms[d], 0, 0, 1, 1); 
    }
   //___________________________________________________
    // Computing first solution data for initializing collection of tensor data

    advanceStokes(
            umac, pres,              /* LHS */
            mfluxdiv, source_terms,  /* RHS */
            alpha_fc, beta, gamma, beta_ed, geom, dt
            );

    //____________________________________________________________________
    // Compute derivatives so that we can find dimensions of the corresponding multifabs
    /* Test pressure gradient calculations (Will need dims below) */
    std::array< MultiFab, AMREX_SPACEDIM > pg;
    for (int d=0; d<AMREX_SPACEDIM; d++)
    {
        pg[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
    }
    SetPressureBC(pres, geom);
    ComputeGrad(pres, pg, 0, 0, 1, 0, geom);
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        pg[d].setVal(0);
    } 

    /* Test div(F) used in NN calculations  (Will need dims below)  */
    MultiFab DivF(ba, dmap, 1, 1); 
    DivF.setVal(0);
    ComputeDiv(DivF,source_terms,0,0,1,geom,0);



    //____________________________________________________________________
    //Compute dims of quantities to be converted to PyTorch tensors

    /* Compute dimensions of single component pressure box */
    const auto & presbox  =   pres[0];
    IntVect presTensordim = presbox.bigEnd()-presbox.smallEnd()+1;

    /* Compute dimensions of single component div(F) box */
    const auto & DivFbox  =   DivF[0];
    IntVect DivFdim = DivFbox.bigEnd()-DivFbox.smallEnd()+1;


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

    /* Compute dimensions of each source pressure gradient box */
    std::vector<int> pgTensordims(9);
    const auto & pgXbox  =   pg[0][0];
    const auto & pgYbox  =   pg[1][0];
    const auto & pgZbox  =   pg[2][0];
    IntVect pgXTensordim = pgXbox.bigEnd()-pgXbox.smallEnd();
    IntVect pgYTensordim = pgYbox.bigEnd()-pgYbox.smallEnd();
    IntVect pgZTensordim = pgZbox.bigEnd()-pgZbox.smallEnd();
    for (int i=0; i<3;++i)
    {
        pgTensordims[i  ]=pgXTensordim[i]+1;
        pgTensordims[i+3]=pgYTensordim[i]+1;
        pgTensordims[i+6]=pgZTensordim[i]+1;
    }


    //____________________________________________________________________
    /* Set Pytorch CUDA device */
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())  device = torch::Device(torch::kCUDA);


    //____________________________________________________________________
    /* Define CNN models and move to GPU and set random seed */
    torch::manual_seed(0);
    auto CNN_UX= std::make_shared<StokesCNNet_Ux>();
    auto CNN_UY= std::make_shared<StokesCNNet_Uy>();
    auto CNN_UZ= std::make_shared<StokesCNNet_Uz>();
    auto CNN_P= std::make_shared<StokesCNNet_P>();
    CNN_UX->to(device);
    CNN_UY->to(device);
    CNN_UZ->to(device);
    CNN_P->to(device);




    //____________________________________________________________________
    /* pointer  to advanceStokes functions in src_hydro/advance.cpp  */
    void (*advanceStokesPtr)(std::array< MultiFab, AMREX_SPACEDIM >&,MultiFab&, 
                                const std::array< MultiFab,AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&,
                                std::array< MultiFab, AMREX_SPACEDIM >&, 
                                MultiFab&,MultiFab&,std::array< MultiFab, NUM_EDGE >&,
                                const Geometry,const Real&
                                ) = &advanceStokes;

    //____________________________________________________________________
    /* Wrap advanceStokes function pointer */
    bool RefineSol=false;
    auto advanceStokes_ML=Wrapper(advanceStokesPtr,RefineSol,device,
                                CNN_UX,CNN_UY,CNN_UZ,CNN_P,
                                presTensordim,sourceTermTensordims,
                                umacTensordims,DivFdim,pgTensordims,dmap,ba);

    RefineSol=true;
    auto advanceStokes_ML2=Wrapper(advanceStokesPtr,RefineSol,device,
                                    CNN_UX,CNN_UY,CNN_UZ,CNN_P,
                                    presTensordim,sourceTermTensordims,
                                    umacTensordims,DivFdim,pgTensordims,dmap,ba) ;




    //____________________________________________________________________
    /* Initialize tensors that collect all pressure and source term data*/
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 

    torch::Tensor presCollect= torch::zeros({1,presTensordim[0], presTensordim[1],presTensordim[2]},options);
    ConvertToTensor(pres,presCollect);

    torch::Tensor DivFCollect= torch::zeros({1,DivFdim[0], DivFdim[1],DivFdim[2]},options);
    ConvertToTensor(DivF,DivFCollect);


    std::array<torch::Tensor,AMREX_SPACEDIM> RHSCollect;
    RHSCollect[0]=torch::zeros({1,sourceTermTensordims[0] , sourceTermTensordims[1],sourceTermTensordims[2] },options);
    RHSCollect[1]=torch::zeros({1,sourceTermTensordims[3] , sourceTermTensordims[4],sourceTermTensordims[5] },options);
    RHSCollect[2]=torch::zeros({1,sourceTermTensordims[6] , sourceTermTensordims[7],sourceTermTensordims[8] },options);
    setVal(source_termsTrimmed, 0.);
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        MultiFab::Copy(source_termsTrimmed[d], source_terms[d], 0, 0, 1, 1); 
    }
    Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSCollect);



    std::array<torch::Tensor,AMREX_SPACEDIM> umacCollect;
    umacCollect[0]=torch::zeros({1,umacTensordims[0] , umacTensordims[1],umacTensordims[2] },options);
    umacCollect[1]=torch::zeros({1,umacTensordims[3] , umacTensordims[4],umacTensordims[5] },options);
    umacCollect[2]=torch::zeros({1,umacTensordims[6] , umacTensordims[7],umacTensordims[8] },options);
    Convert_StdArrMF_To_StdArrTensor(umac,umacCollect);


    std::array<torch::Tensor,AMREX_SPACEDIM> pgCollect;
    pgCollect[0]=torch::zeros({1,pgTensordims[0] , pgTensordims[1],pgTensordims[2] },options);
    pgCollect[1]=torch::zeros({1,pgTensordims[3] , pgTensordims[4],pgTensordims[5] },options);
    pgCollect[2]=torch::zeros({1,pgTensordims[6] , pgTensordims[7],pgTensordims[8] },options);
    Convert_StdArrMF_To_StdArrTensor(pg,pgCollect);


    std::vector<double> TimeDataWindow(50);
    std::vector<double> ResidDataWindow(50);


    //____________________________________________________________________
    // Create copies we will use to compate direct GMRES
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


    /****************************************************************************
     *                                                                          *
     * Advance Time Steps                                                       *
     *                                                                          *
     ***************************************************************************/


    //___________________________________________________________________________
    step = 1;
    int initNum     =64; //number of data points needed before making predictions. Note, must be set in wrapper as well.
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
                mark.pos(0)=0.85*dis(gen);
                mark.pos(1)=0.85*dis(gen);
                mark.pos(2)=0.85*dis(gen);
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

        Print() << "*** COARSE SOLUTION  ***" << "\n"; 
        // // mimicing split in NN-wrapper (for comparison)
        // if (step> initNum)
        // {
        //     gmres_max_iter = 5 ;
        //     advanceStokes(
        //             umacDirect, presDirect,              /* LHS */
        //             mfluxdiv, source_termsDirect,  /* RHS */
        //             alpha_fc, beta, gamma, beta_ed, geom, dt
        //         );
        //     gmres_max_iter = 100 ;
        // }
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


        Print() << "*** COARSE SOLUTION (ML) ***" << "\n"; 
        Real step_strt_time = ParallelDescriptor::second();
        advanceStokes_ML(umac,pres, /* LHS */
                        mfluxdiv,source_terms, /* RHS*/
                        alpha_fc, beta, gamma, beta_ed, geom, dt,
                        presCollect,RHSCollect,umacCollect,DivFCollect,
                        pgCollect, /* ML */
                        step,TimeDataWindow,ResidDataWindow /* ML */
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




        Print() << "*** REFINE SOLUTION (ML) ***" << "\n"; 
        gmres::gmres_abs_tol = 1e-6;
        advanceStokes_ML2(umac,pres, /* LHS */
                        mfluxdiv,source_terms,/* RHS */
                        alpha_fc, beta, gamma, beta_ed, geom, dt,
                        presCollect,RHSCollect,umacCollect,DivFCollect,
                        pgCollect, /* ML */
                        step,TimeDataWindow,ResidDataWindow /* ML */
                        );

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
