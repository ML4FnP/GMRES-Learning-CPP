#ifndef __AMREX_INTERFACES_H
#define __AMREX_INTERFACES_H


#include <array>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <torch/torch.h>

#include "amrex_interfaces.h"



/* copy values of a multifab to Pytorch Tensor  */
template<typename T_src>
void ConvertToTensor(const T_src & mf_in, torch::Tensor & tensor_out) {

    const amrex::BoxArray & ba            = mf_in.boxArray();
    const amrex::DistributionMapping & dm = mf_in.DistributionMap();
          int ncomp                       = mf_in.nComp();
          int ngrow                       = mf_in.nGrow();

    // TODO: Use the new AMReX device-agnostic syntax to achieve the same thing
    // TODO: What to do when we include more than 1 box?

    int i,j,k;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for(amrex::MFIter mfi(mf_in, true); mfi.isValid(); ++ mfi) {
        const auto & in_tile  = mf_in[mfi];

        for(amrex::BoxIterator bit(mfi.growntilebox()); bit.ok(); ++ bit) {
            i=int(bit()[0]) - int(in_tile.smallEnd()[0]);
            j=int(bit()[1]) - int(in_tile.smallEnd()[1]);
            k=int(bit()[2]) - int(in_tile.smallEnd()[2]);
            tensor_out.index({0,i,j,k}) = in_tile(bit());
        }
    }
}



/* copy values of  Pytorch Tensor to a single box multifab */
template<typename T_dest>
void TensorToMultifab(torch::Tensor tensor_in, T_dest & mf_out) {

    const amrex::BoxArray & ba            = mf_out.boxArray();
    const amrex::DistributionMapping & dm = mf_out.DistributionMap();
          int ncomp                       = mf_out.nComp();
          int ngrow                       = mf_out.nGrow();

    // TODO: Use the new AMReX device-agnostic syntax to achieve the same thing
    // TODO: What to do when we include more than 1 box?

    int i,j,k;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for(amrex::MFIter mfi(mf_out, true); mfi.isValid(); ++ mfi) {
        auto & out_tile  = mf_out[mfi];

        for(amrex::BoxIterator bit(mfi.growntilebox()); bit.ok(); ++ bit) {
            i = bit()[0]-out_tile.smallEnd()[0];
            j = bit()[1]-out_tile.smallEnd()[1];
            k = bit()[2]-out_tile.smallEnd()[2];
            out_tile(bit()) = tensor_in.index({0,i,j,k}).item<double>();
        }
    }
}



void TrimSourceMultiFab(
        const std::array<amrex::MultiFab, AMREX_SPACEDIM> & sourceTerms,
              std::array<amrex::MultiFab, AMREX_SPACEDIM> & source_termsTrimmed
    );



void Convert_StdArrMF_To_StdArrTensor(
        const std::array<amrex::MultiFab, AMREX_SPACEDIM> & StdArrMF,
              std::array<torch::Tensor,   AMREX_SPACEDIM> & tensor_out
    );



void Convert_StdArrTensor_To_StdArrMF(
        const std::array<torch::Tensor,   AMREX_SPACEDIM> & tensor_in,
              std::array<amrex::MultiFab, AMREX_SPACEDIM> & mf_out
    );



void CollectScalar(
        const torch::Tensor & tensor_in,
              torch::Tensor & tensor_collect
    );



void CollectMAC(
        const std::array<torch::Tensor, AMREX_SPACEDIM> & mac_tensor_in,
              std::array<torch::Tensor, AMREX_SPACEDIM> & mac_tensor_collect
    );

#endif
