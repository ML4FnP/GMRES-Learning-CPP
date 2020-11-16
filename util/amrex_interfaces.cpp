#include <array>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <torch/torch.h>

#include "amrex_interfaces.h"



//TODO: no need to call it "Source"MultiFab
void TrimSourceMultiFab(
        const std::array<amrex::MultiFab, AMREX_SPACEDIM> & sourceTerms,
              std::array<amrex::MultiFab, AMREX_SPACEDIM> & source_termsTrimmed
    ) {

    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        const amrex::BoxArray & ba = sourceTerms[d].boxArray();
        const amrex::DistributionMapping & dm = sourceTerms[d].DistributionMap();
        source_termsTrimmed[d].define(ba, dm, 1, 1);
        amrex::MultiFab::Copy(source_termsTrimmed[d], sourceTerms[d], 0, 0, 1, 1);
    }
}



// TODO "StdArr" -> "MAC"
/* copy values of single box std:Array MultiFab to Pytorch Tensor  */
void Convert_StdArrMF_To_StdArrTensor(
        const std::array<amrex::MultiFab, AMREX_SPACEDIM> & StdArrMF,
              std::array<torch::Tensor,   AMREX_SPACEDIM> & tensor_out
    ) {

    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        ConvertToTensor(StdArrMF[d], tensor_out[d]);
    }
}



// TODO "StdArr" -> "MAC"
/* copy values of  std::array of Pytorch Tensors to std::array of single box multifabs */
void Convert_StdArrTensor_To_StdArrMF(
        const std::array<torch::Tensor,   AMREX_SPACEDIM> & tensor_in,
              std::array<amrex::MultiFab, AMREX_SPACEDIM> & mf_out
    ) {

    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        TensorToMultifab(tensor_in[d], mf_out[d]);
    }
}



void CollectScalar(
        const torch::Tensor & tensor_in,
              torch::Tensor & tensor_collect
    ) {

    tensor_collect = torch::cat({tensor_collect, tensor_in}, 0);
}



void CollectMAC(
        const std::array<torch::Tensor, AMREX_SPACEDIM> & mac_tensor_in,
              std::array<torch::Tensor, AMREX_SPACEDIM> & mac_tensor_collect
    ) {

    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        mac_tensor_collect[d]=torch::cat({mac_tensor_collect[d], mac_tensor_in[d]}, 0);
    }
}
