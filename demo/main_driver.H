#ifndef _main_driver_H_
#define _main_driver_H_

#include "common_functions.H"
#include "gmres_functions.H"
#include "IBMarkerContainer.H"

#include "arg_pack.h"
#include "wrapper.h"

#include <torch/torch.h>


//_______________________________________________________________________________
// Specify torch interoperability:
// * Define neural network architecture
// * Torch data sets
// * Tensor/MultiFAB interoperability
//

// Ideal to explicitly use std::shared_ptr<MyModule> rather than a
// "TorchModule". Need to set up NN using this approach where the  module is
// registered and constructed in the initializer list. Can also instead first
// construct the holder with a null pointer and then assign it in the
// constructor
struct Net : torch::nn::Module {
    Net(int64_t DimInFlat, int64_t DimOutFlat)
        : linear(
            register_module("linear",
                torch::nn::Linear(
                    torch::nn::LinearOptions(DimInFlat, DimOutFlat).bias(false)
                )
            )
        ) {}

    torch::Tensor forward(
            torch::Tensor x,
            const std::vector<int> SrcTermDims,
            const amrex::IntVect   presTensordim,
            const std::vector<int> umacTensordims
        );

    torch::nn::Linear linear;
};



//-------------------------------------------------------------------------------

//_______________________________________________________________________________
// ML Wrapper for advanceStokes
//


template<typename Signature>
class MLAdvanceStokes;



template<typename F, typename ... Args>
class MLAdvanceStokes<F(Args ...)> : protected Wrapper<F(Args ...)> {
public:

    template<typename Function>
    MLAdvanceStokes(
            Function a_func,
            bool a_RefineSol,
            torch::Device a_device,
            std::shared_ptr<Net> a_NETPres,
            const IntVect a_presTensordim,
            const std::vector<int> a_srctermTensordim,
            const std::vector<int> a_umacTensordims,
            DistributionMapping & a_dmap, BoxArray & a_ba,
            int n_TimeDataWindow
        )
    : Wrapper<F(Args...)>(a_func),
      device(a_device),
      NETPres(a_NETPres),
      presTensordim(a_presTensordim),
      srctermTensordim(a_srctermTensordim),
      umacTensordims(a_umacTensordims),
      dmap(a_dmap),
      ba(a_ba),
      step(0),
      TimeDataWindow(n_TimeDataWindow)
    {}

    F operator()(Args ... args);

    void set_step(int a_step) { step = a_step; }

private:
    bool RefineSol;
    torch::Device device;
    std::shared_ptr<Net> NETPres;  // TODO: make this class the owner of NETPres
    const IntVect presTensordim;
    const std::vector<int> srctermTensordim;
    const std::vector<int> umacTensordims;
    DistributionMapping dmap;
    BoxArray ba;
    int step;
    std::vector<double> TimeDataWindow;
};





//_______________________________________________________________________________
// Abbreviations:
using TileIndex       = IBMarkerContainer::TileIndex;
using ParticleType    = IBMarkerContainer::ParticleType;
using AoS             = IBMarkerContainer::AoS;




/********************************************************************************
 *                                                                              *
 * Output data                                                                  *
 *   - in WritePlotFile.cpp                                                     *
 *                                                                              *
 *******************************************************************************/

void WritePlotFile(const int step,
                   const Real time,
                   const Geometry geom,
                   std::array< MultiFab, AMREX_SPACEDIM >& umac,
                   const MultiFab& pres,
                   const IBMarkerContainer & ib_pc);

///////////////////////////

#endif
