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
    :  convf11(torch::nn::Conv3dOptions(1 , 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf12(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf13(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf21(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf22(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf23(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf24(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf25(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf26(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf31(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf32(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf33(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf34(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf35(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf36(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

    //    convf41(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf42(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf43(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf44(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf45(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf46(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros))
        LRELU(torch::nn::LeakyReLUOptions().negative_slope(0.01))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);

        register_module("convf21", convf21);
        register_module("convf22", convf22);
        register_module("convf23", convf23);
        register_module("convf24", convf24);
        register_module("convf25", convf25);
        register_module("convf26", convf26);

        register_module("convf31", convf31);
        register_module("convf32", convf32);
        register_module("convf33", convf33);
        register_module("convf34", convf34);
        register_module("convf35", convf35);
        register_module("convf36", convf36);

        register_module("LRELU", LRELU);

        // register_module("convf41", convf41);
        // register_module("convf42", convf42);
        // register_module("convf43", convf43);
        // register_module("convf44", convf44);
        // register_module("convf45", convf45);
        // register_module("convf46", convf46);
    }

   torch::Tensor forward(torch::Tensor f1,torch::Tensor f2,torch::Tensor f3,
                            int maxDim,const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
   {
        int64_t Current_batchsize= f1.size(0);

        f1 = LRELU(convf11(f1.unsqueeze(1)));
        f1 = LRELU(convf12(f1));
        f1 = LRELU(convf13(f1));
        f1 = LRELU(convf14(f1));
        f1 = LRELU(convf15(f1));
        f1 = (convf16(f1));
        f1 = f1.squeeze(1);
        f1 = torch::nn::functional::pad(f1, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(0));


        f2 = LRELU(convf21(f2.unsqueeze(1)));
        f2 = LRELU(convf22(f2));
        f2 = LRELU(convf23(f2));
        f2 = LRELU(convf24(f2));
        f2 = LRELU(convf25(f2));
        f2 = (convf26(f2));
        f2 = f2.squeeze(1);
        f2 = torch::nn::functional::pad(f2, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(0));


        f3 = LRELU(convf31(f3.unsqueeze(1)));
        f3 = LRELU(convf32(f3));
        f3 = LRELU(convf33(f3));
        f3 = LRELU(convf34(f3));
        f3 = LRELU(convf35(f3));
        f3 = (convf36(f3));
        f3 = f3.squeeze(1);
        f3 = torch::nn::functional::pad(f3, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(0));


        // Note: network outputs are paded with zeros so that all outputs have the same uniform length
        // of the maximal dimension. This allows for easy addition of the network outputs.
        // The padding is done according to the dimensions of the input source terms 
        torch:: Tensor Ux = f1.index({Slice(),Slice(0,umacTensordims[0]),Slice(0,umacTensordims[1]),Slice(0,umacTensordims[2])})
                            + f2.index({Slice(),Slice(0,umacTensordims[0]),Slice(0,umacTensordims[1]),Slice(0,umacTensordims[2])})
                            + f3.index({Slice(),Slice(0,umacTensordims[0]),Slice(0,umacTensordims[1]),Slice(0,umacTensordims[2])});


    //   Ux = torch::relu(convf41(Ux.unsqueeze(1)));
    //   Ux = torch::relu(convf42(Ux));
    //   Ux = torch::relu(convf43(Ux));
    //   Ux = torch::relu(convf44(Ux));
    //   Ux = torch::relu(convf45(Ux));
    //   Ux = convf46(Ux);
    //   Ux = Ux.squeeze(1);

        return Ux;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16;
   torch::nn::Conv3d convf21,convf22,convf23,convf24,convf25,convf26;
   torch::nn::Conv3d convf31,convf32,convf33,convf34,convf35,convf36;
//    torch::nn::Conv3d convf41,convf42,convf43,convf44,convf45,convf46;
   torch::nn::LeakyReLU LRELU;

};
//////////////////////////////////////////////////////////////////////////////
struct StokesCNNet_Uy : torch::nn::Module {
  StokesCNNet_Uy( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf12(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf13(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf21(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf22(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf23(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf24(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf25(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf26(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf31(torch::nn::Conv3dOptions(1 ,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf32(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf33(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf34(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf35(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf36(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros))

    //    convf41(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf42(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf43(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf44(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf45(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
    //    convf46(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);

        register_module("convf21", convf21);
        register_module("convf22", convf22);
        register_module("convf23", convf23);
        register_module("convf24", convf24);
        register_module("convf25", convf25);
        register_module("convf26", convf26);

        register_module("convf31", convf31);
        register_module("convf32", convf32);
        register_module("convf33", convf33);
        register_module("convf34", convf34);
        register_module("convf35", convf35);
        register_module("convf36", convf36);

        // register_module("convf41", convf41);
        // register_module("convf42", convf42);
        // register_module("convf43", convf43);
        // register_module("convf44", convf44);
        // register_module("convf45", convf45);
        // register_module("convf46", convf46);
    }

   torch::Tensor forward(torch::Tensor f1,torch::Tensor f2,torch::Tensor f3,
                            int maxDim,const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
   {
        int64_t Current_batchsize= f1.size(0);

        f1 = torch::relu(convf11(f1.unsqueeze(1)));
        f1 = torch::relu(convf12(f1));
        f1 = torch::relu(convf13(f1));
        f1 = torch::relu(convf14(f1));
        f1 = torch::relu(convf15(f1));
        f1 = (convf16(f1));
        f1 = f1.squeeze(1);
        f1 = torch::nn::functional::pad(f1, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(0));


        f2 = torch::relu(convf21(f2.unsqueeze(1)));
        f2 = torch::relu(convf22(f2));
        f2 = torch::relu(convf23(f2));
        f2 = torch::relu(convf24(f2));
        f2 = torch::relu(convf25(f2));
        f2 = (convf26(f2));
        f2 = f2.squeeze(1);
        f2 = torch::nn::functional::pad(f2, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(0));


        f3 = torch::relu(convf31(f3.unsqueeze(1)));
        f3 = torch::relu(convf32(f3));
        f3 = torch::relu(convf33(f3));
        f3 = torch::relu(convf34(f3));
        f3 = torch::relu(convf35(f3));
        f3 = (convf36(f3));
        f3 = f3.squeeze(1);
        f3 = torch::nn::functional::pad(f3, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(0));

        torch:: Tensor Uy =   f1.index({Slice(),Slice(0,umacTensordims[3]),Slice(0,umacTensordims[4]),Slice(0,umacTensordims[5])})
                            + f2.index({Slice(),Slice(0,umacTensordims[3]),Slice(0,umacTensordims[4]),Slice(0,umacTensordims[5])})
                            + f3.index({Slice(),Slice(0,umacTensordims[3]),Slice(0,umacTensordims[4]),Slice(0,umacTensordims[5])});

    //   Uy = torch::relu(convf41(Uy.unsqueeze(1)));
    //   Uy = torch::relu(convf42(Uy));
    //   Uy = torch::relu(convf43(Uy));
    //   Uy = torch::relu(convf44(Uy));
    //   Uy = torch::relu(convf45(Uy));
    //   Uy = convf46(Uy);
    //   Uy = Uy.squeeze(1);

        return Uy;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16;
   torch::nn::Conv3d convf21,convf22,convf23,convf24,convf25,convf26;
   torch::nn::Conv3d convf31,convf32,convf33,convf34,convf35,convf36;
//    torch::nn::Conv3d convf41,convf42,convf43,convf44,convf45,convf46;
};
//////////////////////////////////////////////////////////////////////////////
struct StokesCNNet_Uz : torch::nn::Module {
  StokesCNNet_Uz( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf12(torch::nn::Conv3dOptions(1, 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf13(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf14(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf15(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf16(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf21(torch::nn::Conv3dOptions(1 , 1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf22(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf23(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf24(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf25(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf26(torch::nn::Conv3dOptions(1, 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),

       convf31(torch::nn::Conv3dOptions(1 ,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf32(torch::nn::Conv3dOptions(1,1, 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf33(torch::nn::Conv3dOptions(1,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf34(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf35(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros)),
       convf36(torch::nn::Conv3dOptions(1 ,1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(false).padding_mode(torch::kZeros))
    { 
        register_module("convf11", convf11);
        register_module("convf12", convf12);
        register_module("convf13", convf13);
        register_module("convf14", convf14);
        register_module("convf15", convf15);
        register_module("convf16", convf16);

        register_module("convf21", convf21);
        register_module("convf22", convf22);
        register_module("convf23", convf23);
        register_module("convf24", convf24);
        register_module("convf25", convf25);
        register_module("convf26", convf26);

        register_module("convf31", convf31);
        register_module("convf32", convf32);
        register_module("convf33", convf33);
        register_module("convf34", convf34);
        register_module("convf35", convf35);
        register_module("convf36", convf36);
    }

   torch::Tensor forward(torch::Tensor f1,torch::Tensor f2,torch::Tensor f3,
                            int maxDim,const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
   {
        int64_t Current_batchsize= f1.size(0);

        f1 = torch::relu(convf11(f1.unsqueeze(1)));
        f1 = torch::relu(convf12(f1));
        f1 = torch::relu(convf13(f1));
        f1 = torch::relu(convf14(f1));
        f1 = torch::relu(convf15(f1));
        f1 = (convf16(f1));
        f1 = f1.squeeze(1);
        f1 = torch::nn::functional::pad(f1, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(0));


        f2 = torch::relu(convf21(f2.unsqueeze(1)));
        f2 = torch::relu(convf22(f2));
        f2 = torch::relu(convf23(f2));
        f2 = torch::relu(convf24(f2));
        f2 = torch::relu(convf25(f2));
        f2 = (convf26(f2));
        f2 = f2.squeeze(1);
        f2 = torch::nn::functional::pad(f2, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(0));


        f3 = torch::relu(convf31(f3.unsqueeze(1)));
        f3 = torch::relu(convf32(f3));
        f3 = torch::relu(convf33(f3));
        f3 = torch::relu(convf34(f3));
        f3 = torch::relu(convf35(f3));
        f3 = (convf36(f3));
        f3 = f3.squeeze(1);
        f3 = torch::nn::functional::pad(f3, torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(0));

        torch:: Tensor Uz =   f1.index({Slice(),Slice(0,umacTensordims[6]),Slice(0,umacTensordims[7]),Slice(0,umacTensordims[8])})
                            + f2.index({Slice(),Slice(0,umacTensordims[6]),Slice(0,umacTensordims[7]),Slice(0,umacTensordims[8])})
                            + f3.index({Slice(),Slice(0,umacTensordims[6]),Slice(0,umacTensordims[7]),Slice(0,umacTensordims[8])});

        return Uz;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16;
   torch::nn::Conv3d convf21,convf22,convf23,convf24,convf25,convf26;
   torch::nn::Conv3d convf31,convf32,convf33,convf34,convf35,convf36;
};

////////////////////////////////////////////////////////////////////////////////
/* Pressure CNN */
struct StokesCNNet_P : torch::nn::Module {
  StokesCNNet_P( )
    :  convf11(torch::nn::Conv3dOptions(1 , 1,  3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf12(torch::nn::Conv3dOptions(1 , 1,  3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
       convf13(torch::nn::Conv3dOptions(1 , 1 , 3).stride(1).padding(3).dilation(3).groups(1).bias(true).padding_mode(torch::kCircular)),
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

   torch::Tensor forward(torch::Tensor divF, const IntVect presTensordim)
   {
        int64_t Current_batchsize= divF.size(0);
        divF = LRELU(BNorm3D1(convf11(divF.unsqueeze(1))));
        divF = LRELU(BNorm3D2(convf12(divF)));
        divF = LRELU(BNorm3D3(convf13(divF)));
        divF = LRELU(BNorm3D4(convf14(divF)));
        divF = LRELU((convf15(divF)));
        divF = LRELU((convf16(divF)));
        divF = LRELU((convf17(divF)));
        divF = LRELU((convf18(divF)));
        divF = ((convf19(divF)));
        torch::Tensor P = divF.squeeze(1);
        return P;
   }
   torch::nn::Conv3d convf11,convf12,convf13,convf14,convf15,convf16,convf17,convf18,convf19;
   torch::nn::LeakyReLU LRELU;
   torch::nn::BatchNorm3d BNorm3D1,BNorm3D2,BNorm3D3,BNorm3D4,BNorm3D5;

};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/* Need to define a classes to use the Pytorch data loader */

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

        

            // The tensors are padded so that every component is the same size as the largest component
            // Note: This allows the tensors to be concatencated, and allows the most direct use of the pytorch dataloader
            int maxP   = *std::max_element(presTensordim.begin(), presTensordim.end());
            int maxU   = *std::max_element(umacTensordims.begin(), umacTensordims.end());
            int maxSrc = *std::max_element(srctermTensordim.begin(), srctermTensordim.end());
            int max1   = std::max(maxP,maxU);
            int maxDim    = std::max(max1,maxSrc);
            
            SrcTensor[0]= torch::nn::functional::pad(SrcTensor[0], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(0));
            SrcTensor[1]= torch::nn::functional::pad(SrcTensor[1], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(0));
            SrcTensor[2]= torch::nn::functional::pad(SrcTensor[2], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(0));

            umacTensors[0]= torch::nn::functional::pad(umacTensors[0], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[2], 0, maxDim-umacTensordims[1],0, maxDim-umacTensordims[0]}).mode(torch::kConstant).value(0));
            umacTensors[1]= torch::nn::functional::pad(umacTensors[1], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[5], 0, maxDim-umacTensordims[4],0, maxDim-umacTensordims[3]}).mode(torch::kConstant).value(0));
            umacTensors[2]= torch::nn::functional::pad(umacTensors[2], torch::nn::functional::PadFuncOptions({0, maxDim-umacTensordims[8], 0, maxDim-umacTensordims[7],0, maxDim-umacTensordims[6]}).mode(torch::kConstant).value(0));

            Pres= torch::nn::functional::pad(Pres, torch::nn::functional::PadFuncOptions({0, maxDim-presTensordim[2], 0, maxDim-presTensordim[1],0,maxDim-presTensordim[0]}).mode(torch::kConstant).value(0));


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
//////////////////////////////////////////////////////////////////////////////
/* Dataset class for pressure CNN */
class CustomDatasetCNN_Pres : public torch::data::Dataset<CustomDatasetCNN_Pres>
{
    private:
        torch::Tensor bTensor, SolTensor;
        torch::Tensor Pres;
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow)
{  
    return  umacCollect;
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,torch::Tensor& DivFCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,torch::Tensor DivFfinal)
{  

        DivFCollect=torch::cat({DivFCollect,DivFfinal},0);
    
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
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
                   std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect, torch::Tensor& DivFCollect
                   ,int step,std::vector<double>& TimeDataWindow,std::vector<double>& ResidDataWindow
                   ,amrex::DistributionMapping dmap, BoxArray  ba, std::array<MultiFab, AMREX_SPACEDIM>& source_termsTrimmed)
{  
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
        MultiFab::Copy(source_termsTrimmed[d], sourceTerms[d], 0, 0, 1, 1); 
    }
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void  CNN_TrainLoop(std::shared_ptr<StokesCNNet_Ux> CNN_UX,std::shared_ptr<StokesCNNet_Uy> CNN_UY,
                       std::shared_ptr<StokesCNNet_Uz> CNN_UZ,
                       std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,torch::Tensor& PresCollect,
                       std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,const IntVect presTensordim, 
                       const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
{
                /*Setting up learning loop below */
                torch::optim::SGD optimizerUx({CNN_UX->parameters()}, torch::optim::SGDOptions(0.001));
                torch::optim::SGD optimizerUy({CNN_UY->parameters()}, torch::optim::SGDOptions (0.001));
                torch::optim::SGD optimizerUz({CNN_UZ->parameters()}, torch::optim::SGDOptions (0.001));

                // torch::optim::Adagrad optimizerUx({CNN_UX->parameters()}, torch::optim::AdagradOptions(0.01));
                // torch::optim::Adagrad optimizerUy({CNN_UY->parameters()}, torch::optim::AdagradOptions (0.01));
                // torch::optim::Adagrad optimizerUz({CNN_UZ->parameters()}, torch::optim::AdagradOptions (0.01));
                // torch::optim::Adagrad optimizerPres({CNN_P->parameters()}, torch::optim::AdagradOptions (0.01));



                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN(RHSCollect,PresCollect,umacCollect,presTensordim, srctermTensordim,umacTensordims).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 250; 

                float lossUx = 0.0;
                float lossUy = 0.0;
                float lossUz = 10.0;

                // Compute maximal dimension for slicing down tensor output of dataloader object
                int maxU   = *std::max_element(umacTensordims.begin(), umacTensordims.end());
                int maxSrc = *std::max_element(srctermTensordim.begin(), srctermTensordim.end());
                int maxDim    = std::max(maxU,maxSrc);

               
                // Set indicies for removing the padded elements of the tensors used in the dataloader
                // Note: To use the pytorch dataloader(which makes it easy to use the autograd feature)
                // we must set tensor data pairs (target and data) as an output. 
                // To do this, the source,umac,and pressure Multifabs
                // that are converted to tensors are first concatenated by pading them 
                // so that every dimension has the same size as the largest dimension.
                // Below, tensor indicies are set so we remove this padding
                auto f1_Slice_x =Slice(0,srctermTensordim[0]-maxDim);
                auto f1_Slice_y =Slice(0,srctermTensordim[1]-maxDim);
                auto f1_Slice_z =Slice(0,srctermTensordim[2]-maxDim);
                auto f2_Slice_x =Slice(0,srctermTensordim[3]-maxDim);
                auto f2_Slice_y =Slice(0,srctermTensordim[4]-maxDim);
                auto f2_Slice_z =Slice(0,srctermTensordim[5]-maxDim);
                auto f3_Slice_x =Slice(0,srctermTensordim[6]-maxDim);
                auto f3_Slice_y =Slice(0,srctermTensordim[7]-maxDim);
                auto f3_Slice_z =Slice(0,srctermTensordim[8]-maxDim);

                if (srctermTensordim[0]-maxDim==0) f1_Slice_x =Slice();
                if (srctermTensordim[1]-maxDim==0) f1_Slice_y =Slice();
                if (srctermTensordim[2]-maxDim==0) f1_Slice_z =Slice();
                if (srctermTensordim[3]-maxDim==0) f2_Slice_x =Slice();
                if (srctermTensordim[4]-maxDim==0) f2_Slice_y =Slice();
                if (srctermTensordim[5]-maxDim==0) f2_Slice_z =Slice();
                if (srctermTensordim[6]-maxDim==0) f3_Slice_x =Slice();
                if (srctermTensordim[7]-maxDim==0) f3_Slice_y =Slice();
                if (srctermTensordim[8]-maxDim==0) f3_Slice_z =Slice();
                

                /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
                dataset, type of sampler, etc */
                auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches

                /* NOTE: Weights and tensors must be of the same type. So, for function calls where input tensors interact with
                        weights, the input tensors are converted to  single precesion(weights are floats by default). Alternatively we can 
                        convert weights to double precision (at the cost of performance) */
                while((lossUx+lossUy+lossUz)>5*e1 and epoch<numEpoch )
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

                            // forward pass
                            // Note: Inputs to forward function are de-paded appropriately
                            // NOTE: The padded outputs of the CNN are de-padded and then  added together as the model output argument of the loss functions
                            torch::Tensor outputUx = CNN_UX->forward(data.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                                maxDim,srctermTensordim,umacTensordims);

                            torch::Tensor outputUy = CNN_UY->forward(data.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                                maxDim,srctermTensordim,umacTensordims);
                                                
                            torch::Tensor outputUz = CNN_UZ->forward(data.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                                data.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                                maxDim,srctermTensordim,umacTensordims);
        


                            //evaulate loss
                            // Note: target solution data input to  loss function are also de-paded appropriately (since they are padded prior to being added to the dataloader)
                            // auto loss_out = torch::nn::functional::mse_loss(output,target, torch::nn::functional::MSELossFuncOptions(torch::kSum));

                            torch::Tensor loss_outUx = torch::mse_loss(outputUx,
                                            target.index({Slice(),0,Slice(0,umacTensordims[0]),
                                            Slice(0,umacTensordims[1])
                                            ,Slice(0,umacTensordims[2])}).to(torch::kFloat32));  

                            torch::Tensor loss_outUy = torch::mse_loss(outputUy,
                                            target.index({Slice(),1,Slice(0,umacTensordims[3]),
                                            Slice(0,umacTensordims[4])
                                            ,Slice(0,umacTensordims[5])}).to(torch::kFloat32));  

                            torch::Tensor loss_outUz = torch::mse_loss(outputUz,
                                            target.index({Slice(),2,Slice(0,umacTensordims[6]),
                                            Slice(0,umacTensordims[7])
                                            ,Slice(0,umacTensordims[8])}).to(torch::kFloat32));   

                            torch::Tensor TotalLoss =loss_outUx+loss_outUy+loss_outUz;
                            // Extract loss value for printing to console
                            lossUx = loss_outUx.item<float>();
                            lossUy = loss_outUy.item<float>();
                            lossUz = loss_outUz.item<float>();

                            // Backward pass
                            TotalLoss.backward();
                            // loss_outUx.backward();
                            // loss_outUy.backward();
                            // loss_outUz.backward();

                            // Apply gradients
                            optimizerUx.step();
                            optimizerUy.step();
                            optimizerUz.step();
                            epoch = epoch +1;

                            // Print loop info to console
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss Ux: "  << lossUx << std::endl;
                    std::cout << "Loss Uy: "  << lossUy << std::endl;
                    std::cout << "Loss Uz: "  << lossUz << std::endl;
                    std::cout << "Epoch Number: " << epoch << std::endl;
                }

}

///////////////////////////////////////////////////////////////////////////////////////////
void  CNN_P_TrainLoop( std::shared_ptr<StokesCNNet_P> CNN_P,
                       torch::Tensor& DivSrcTensorCollect,torch::Tensor& PresCollect,
                       const IntVect presTensordim, const IntVect DivFdim)
{
                /*Setting up learning loop below */
                // torch::optim::SGD optimizerPres({CNN_P->parameters()}, torch::optim::SGDOptions (0.01));
                torch::optim::Adagrad optimizerPres({CNN_P->parameters()}, torch::optim::AdagradOptions (0.001));

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDatasetCNN_Pres(DivSrcTensorCollect,PresCollect,DivFdim,presTensordim).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 3000; 

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
                            torch::Tensor outputP = CNN_P->forward(data.to(torch::kFloat32),presTensordim);


                            //evaulate loss                        
                            // auto loss_out = torch::nn::functional::mse_loss(output,target, torch::nn::functional::MSELossFuncOptions(torch::kSum));
                            torch::Tensor loss_outP = torch::mse_loss(outputP,target.to(torch::kFloat32));  

                            // Extract loss value for printing to console
                            lossP  = loss_outP.item<float>();

                            // Backward pass
                            loss_outP.backward();


                            // Apply gradients
                            optimizerPres.step();

                            epoch = epoch +1;

                            // Print loop info to console
                    }
                    std::cout << "___________" << std::endl;
                    std::cout << "Loss P : "  << lossP << std::endl;
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
                 amrex::DistributionMapping dmap, BoxArray  ba)
{
    auto new_function = [func,RefineSol,device,CNN_UX,CNN_UY,CNN_UZ,CNN_P,presTensordim,srctermTensordim,umacTensordims,DivFdim,dmap,ba](auto&&... args)
    {
        int retrainFreq =64;
        int initNum     =128;
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
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed,RHSTensor); /* Convert Std::array<MultiFab,AMREX_SPACEDIM > to  std::array<torch::tensor, AMREX_SPACEDIM> */


            /* Appropriately Pad input */ 
            /* Add channel dim to ever tensor component of the Std::array<torch::tensor,Amrex_dim> objects */
            for (int d=0;d<AMREX_SPACEDIM;d++) 
            {
                RHSTensor[d]=RHSTensor[d].unsqueeze(1);
            }
            // The tensors are padded so that every component is the same size as the largest component
            // Note: This allows the tensors to be concatencated, and allows the most direct use of the pytorch dataloader
            int maxP   = *std::max_element(presTensordim.begin(), presTensordim.end());
            int maxU   = *std::max_element(umacTensordims.begin(), umacTensordims.end());
            int maxSrc = *std::max_element(srctermTensordim.begin(), srctermTensordim.end());
            int max1   = std::max(maxP,maxU);
            int maxDim    = std::max(max1,maxSrc);
            RHSTensor[0]= torch::nn::functional::pad(RHSTensor[0], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[2], 0, maxDim-srctermTensordim[1],0, maxDim-srctermTensordim[0]}).mode(torch::kConstant).value(0));
            RHSTensor[1]= torch::nn::functional::pad(RHSTensor[1], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[5], 0, maxDim-srctermTensordim[4],0, maxDim-srctermTensordim[3]}).mode(torch::kConstant).value(0));
            RHSTensor[2]= torch::nn::functional::pad(RHSTensor[2], torch::nn::functional::PadFuncOptions({0, maxDim-srctermTensordim[8], 0, maxDim-srctermTensordim[7],0, maxDim-srctermTensordim[6]}).mode(torch::kConstant).value(0));
            torch::Tensor SourceTermTensor  =torch::cat({RHSTensor[0],RHSTensor[1],RHSTensor[2]},1);

            torch::Tensor DivFTensor= torch::zeros({1,DivFdim[0],DivFdim[1],DivFdim[2] },options);
            ConvertToTensor(DivF,DivFTensor);

            // Slicing of source terms
            auto f1_Slice_x =Slice(0,srctermTensordim[0]-maxDim);
            auto f1_Slice_y =Slice(0,srctermTensordim[1]-maxDim);
            auto f1_Slice_z =Slice(0,srctermTensordim[2]-maxDim);
            auto f2_Slice_x =Slice(0,srctermTensordim[3]-maxDim);
            auto f2_Slice_y =Slice(0,srctermTensordim[4]-maxDim);
            auto f2_Slice_z =Slice(0,srctermTensordim[5]-maxDim);
            auto f3_Slice_x =Slice(0,srctermTensordim[6]-maxDim);
            auto f3_Slice_y =Slice(0,srctermTensordim[7]-maxDim);
            auto f3_Slice_z =Slice(0,srctermTensordim[8]-maxDim);
            if (srctermTensordim[0]-maxDim==0) f1_Slice_x =Slice();
            if (srctermTensordim[1]-maxDim==0) f1_Slice_y =Slice();
            if (srctermTensordim[2]-maxDim==0) f1_Slice_z =Slice();
            if (srctermTensordim[3]-maxDim==0) f2_Slice_x =Slice();
            if (srctermTensordim[4]-maxDim==0) f2_Slice_y =Slice();
            if (srctermTensordim[5]-maxDim==0) f2_Slice_z =Slice();
            if (srctermTensordim[6]-maxDim==0) f3_Slice_x =Slice();
            if (srctermTensordim[7]-maxDim==0) f3_Slice_y =Slice();
            if (srctermTensordim[8]-maxDim==0) f3_Slice_z =Slice();

            /* Get prediction as tensor */
            std::array<torch::Tensor,AMREX_SPACEDIM> umacTensor;
            umacTensor[0] = CNN_UX->forward(SourceTermTensor.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                maxDim,srctermTensordim,umacTensordims);

            umacTensor[1] = CNN_UY->forward(SourceTermTensor.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                maxDim,srctermTensordim,umacTensordims);
                                                
            umacTensor[2] = CNN_UZ->forward(SourceTermTensor.index({Slice(),0,f1_Slice_x,f1_Slice_y,f1_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),1,f2_Slice_x,f2_Slice_y,f2_Slice_z}).to(torch::kFloat32),
                                SourceTermTensor.index({Slice(),2,f3_Slice_x,f3_Slice_y,f3_Slice_z}).to(torch::kFloat32),
                                maxDim,srctermTensordim,umacTensordims);

            torch::Tensor presTensor = CNN_P->forward(DivFTensor.to(torch::kFloat32),presTensordim);



            /* Convert tensors to multifab using distribution map of original pressure MultiFab */
            TensorToMultifab(presTensor,presNN);

            /* Convert std::array tensor to std::array multifab using distribution map of original pressure MultiFab */
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

            // gmres_max_iter = 5 ;
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
            // gmres_max_iter = 100;

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

            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);

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
            torch::Tensor presTensor= torch::zeros({1,presTensordim[0] , presTensordim[1],presTensordim[2] },options);
            ConvertToTensor(Unpack_pres(args...),presTensor);
            CollectPressure(args...,presTensor);

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
                CNN_TrainLoop(CNN_UX,CNN_UY,CNN_UZ,
                            Unpack_RHSCollect(args...),Unpack_PresCollect(args...),Unpack_umacCollect(args...),presTensordim,
                            srctermTensordim,umacTensordims);

            /* Train model every time 3 new data points have been added to training set after initialization period */
            }else if ( SampleIndex>0 and SampleIndex%retrainFreq==0 )
            {
                CNN_P_TrainLoop(CNN_P, Unpack_DivFCollect(args...), Unpack_PresCollect(args...),presTensordim, DivFdim);
                CNN_TrainLoop(CNN_UX,CNN_UY,CNN_UZ,
                            Unpack_RHSCollect(args...),Unpack_PresCollect(args...),Unpack_umacCollect(args...),presTensordim,
                            srctermTensordim,umacTensordims);
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


    /* Compute div(F) used in NN calculations */
    MultiFab DivF(ba, dmap, 1, 1); 
    DivF.setVal(0);
    ComputeDiv(DivF,source_terms,0,0,1,geom,0);

    /* Set Pytorch CUDA device */
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())  device = torch::Device(torch::kCUDA);


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


    /* Define CNN models and move to GPU */
    auto CNN_UX= std::make_shared<StokesCNNet_Ux>();
    auto CNN_UY= std::make_shared<StokesCNNet_Uy>();
    auto CNN_UZ= std::make_shared<StokesCNNet_Uz>();
    auto CNN_P= std::make_shared<StokesCNNet_P>();
    CNN_UX->to(device);
    CNN_UY->to(device);
    CNN_UZ->to(device);
    CNN_P->to(device);




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
    auto advanceStokes_ML=Wrapper(advanceStokesPtr,RefineSol,device,
                                CNN_UX,CNN_UY,CNN_UZ,CNN_P,
                                presTensordim,sourceTermTensordims,
                                umacTensordims,DivFdim,dmap,ba);

    RefineSol=true;
    auto advanceStokes_ML2=Wrapper(advanceStokesPtr,RefineSol,device,
                                    CNN_UX,CNN_UY,CNN_UZ,CNN_P,
                                    presTensordim,sourceTermTensordims,
                                    umacTensordims,DivFdim,dmap,ba) ;


    /* Initialize tensors that collect all pressure and source term data*/
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 

    torch::Tensor presCollect= torch::zeros({1,presTensordim[0], presTensordim[1],presTensordim[2]},options);
    torch::Tensor DivFCollect= torch::zeros({1,DivFdim[0], DivFdim[1],DivFdim[2]},options);


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



    // int initNum     =64; //number of data points needed before making predictions. Note, must be set in wrapper as well.
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
                mark.pos(0)=0.95*dis(gen);
                mark.pos(1)=0.95*dis(gen);
                mark.pos(2)=0.95*dis(gen);
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
        // mimicing split in NN-wrapper (for comparison)
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
                        presCollect,RHSCollect,umacCollect,DivFCollect, /* ML */
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
                        presCollect,RHSCollect,umacCollect,DivFCollect, /* ML */
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
