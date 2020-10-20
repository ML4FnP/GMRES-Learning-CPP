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

#include "arg_pack.h"

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
struct Net : torch::nn::Module {
  Net(int64_t DimInFlat, int64_t DimOutFlat)
    : linear(register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(DimInFlat,DimOutFlat).bias(false))))
    { }
   torch::Tensor forward(torch::Tensor x, const std::vector<int> SrcTermDims, const amrex::IntVect presTensordim,const std::vector<int> umacTensordims )
   {

    int64_t Current_batchsize= x.size(0);
    x = linear(x);
    // x = x.index({Slice(),Slice(),Slice(),Slice(-presTensordim[0]*presTensordim[1]*presTensordim[2]-1,-1)});
    // x = x.reshape({Current_batchsize,presTensordim[0],presTensordim[1],presTensordim[2]}); // unflatten

    // Print() << x.size(0) <<  " " << x.size(1) <<  " "  << x.size(2) <<  " " << x.size(3) <<  "\n "; 
    return x;
   }
  torch::nn::Linear linear;
};




/* Need to define a class to use the Pytorch data loader */
class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        torch::Tensor bTensor, SolTensor;

        torch::Tensor SrcTensorFlatx,SrcTensorFlaty,SrcTensorFlatz,SrcTensorFlat;
        torch::Tensor umacFlatx,umacFlaty,umacFlatz,umacFlat;
        torch::Tensor bp,PressureFlat;


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


        //   Print() << SrcTensorFlat.size(0) << " "<< SrcTensorFlat.size(1) << " "<< SrcTensorFlat.size(2) << " "<< SrcTensorFlat.size(3) << "\n "; 
        //   Print() << umacFlat.size(0) << " "<< umacFlat.size(1) << " "<< umacFlat.size(2) << " "<< umacFlat.size(3) << "\n "; 
        // Print() << "Test " << SrcTensorFlat[0].size(0) << " " << SrcTensorFlat[0].size(1)<< " " << SrcTensorFlat[0].size(2) << "\n";





        //   bTensor = bIn[0];
        //   SolTensor = SolIn;

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
/* Functions for unpacking parameter pack passed to wrapped function. */

//______________________________________________________________________________
// Define input arguments used by wrappers -- if you define an arg_pack in the\
// right order, then you can unpack an arg by using: arg_pack::get<arg_NAME>()

enum args {
    arg_umac,
    arg_pres,
    arg_stochMfluxdiv,
    arg_sourceTerms,
    arg_alpha_fc,
    arg_beta,
    arg_gamma,
    arg_beta_ed,
    arg_geom,
    arg_dt,
    ml_PresCollect,
    ml_RHSCollect,
    ml_umacCollect,
    ml_step,
    ml_TimeDataWindow,
    ml_PreFinal
};

// Index of the cfd variables
int farg_cfd = arg_umac;
int larg_cfd = arg_dt;
// Index of the ml variables
int farg_ml = ml_PresCollect;
int larg_ml = ml_PreFinal;


void CollectPressure(std::array< MultiFab, AMREX_SPACEDIM >& umac,MultiFab& pres,
                   const std::array< MultiFab, AMREX_SPACEDIM >& stochMfluxdiv,
                   std::array< MultiFab, AMREX_SPACEDIM >& sourceTerms,
                   std::array< MultiFab, AMREX_SPACEDIM >& alpha_fc,
                   MultiFab& beta,
                   MultiFab& gamma,
                   std::array< MultiFab, NUM_EDGE >& beta_ed,
                   const Geometry geom, const Real& dt,
                   torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow
                   ,torch::Tensor PresFinal)
{
        PresCollect=torch::cat({PresCollect,PresFinal},0);
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
                   ,int step,std::vector<double>& TimeDataWindow)
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
                   torch::Tensor& PresCollect, std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect
                   ,int step,std::vector<double>& TimeDataWindow,
                   std::array<torch::Tensor,AMREX_SPACEDIM> RHSFinal)
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
                   ,int step,std::vector<double>& TimeDataWindow,
                   std::array<torch::Tensor,AMREX_SPACEDIM> umacFinal)
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
                   ,int step,std::vector<double>& TimeDataWindow,
                   std::vector<double>& TimeDataWindowIn)
{
       TimeDataWindow=TimeDataWindowIn;
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
                   ,int step,std::vector<double>& TimeDataWindow,
                   amrex::DistributionMapping dmap, BoxArray  ba, std::array<MultiFab, AMREX_SPACEDIM>& source_termsTrimmed)
{
    for (int d=0; d<AMREX_SPACEDIM; ++d)
    {
        source_termsTrimmed[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 1);
        MultiFab::Copy(source_termsTrimmed[d], sourceTerms[d], 0, 0, 1, 1);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void  TrainLoop(std::shared_ptr<Net> NETPres,std::array<torch::Tensor,AMREX_SPACEDIM>& RHSCollect,torch::Tensor& PresCollect,std::array<torch::Tensor,AMREX_SPACEDIM>& umacCollect,const IntVect presTensordim, const std::vector<int> srctermTensordim, const std::vector<int> umacTensordims)
{
/*Setting up learning loop below */
                torch::optim::Adagrad optimizer(NETPres->parameters(), torch::optim::AdagradOptions(0.01));

                /* Create dataset object from tensors that have collected relevant data */
                auto custom_dataset = CustomDataset(RHSCollect,PresCollect,umacCollect).map(torch::data::transforms::Stack<>());

                int64_t batch_size = 8;
                float e1 = 1e-5;
                int64_t epoch = 0;
                int64_t numEpoch = 500;
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



double MovingAvg (std::vector<double>& TimeDataWindow)
{
    int window = TimeDataWindow.size();
    double sum;
    for(int i = 0 ; i<window ; i++)
    {
        sum+=TimeDataWindow[i];
    }
    sum = sum/double(window);
    return sum;
}




/* ML Wrapper for advanceStokes */
template<typename F>
auto Wrapper(F func,
             bool RefineSol,
             torch::Device device, std::shared_ptr<Net> NETPres,
             const IntVect presTensordim,
             const std::vector<int> srctermTensordim,
             const std::vector<int> umacTensordims,
             amrex::DistributionMapping dmap, BoxArray ba
    ) {

    auto new_function =
    [
        func,
        RefineSol,
        device, NETPres,
        presTensordim,
        srctermTensordim,
        umacTensordims,
        dmap, ba
    ](auto && ... args) {

        // Hard-coded training parameters -- TODO: don't hard-code options
        int retrainFreq = 3;
        int initNum     = 20;

        auto options = torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(torch::kCUDA)
                              .requires_grad(false);

        auto ap = make_arg_pack(args...);

        int                           step = ap.template get<ml_step>();
        std::vector<double> TimeDataWindow = ap.template get<ml_TimeDataWindow>();
        int                      WindowIdx = ((step-initNum)-1)%TimeDataWindow.size();

        // Set initialial guess for pressure and umac to zero -- TODO: is that
        // really a good idea?
        MultiFab pres(ba, dmap, 1, 1);
        pres.setVal(0.);

        std::array< MultiFab, AMREX_SPACEDIM > umac;
        defineFC(umac, ba, dmap, 1);
        setVal(umac, 0.);

        /* Use NN to predict pressure */
        if ((RefineSol == false) && (step > initNum)) {
            //
            // Represent RHS as a std::array
            std::array<torch::Tensor, AMREX_SPACEDIM> RHSTensor;

            RHSTensor[0] = torch::zeros({1,
                                         srctermTensordim[0],
                                         srctermTensordim[1],
                                         srctermTensordim[2]}, options);
            RHSTensor[1] = torch::zeros({1,
                                         srctermTensordim[3],
                                         srctermTensordim[4],
                                         srctermTensordim[5]}, options);
            RHSTensor[2] = torch::zeros({1,
                                         srctermTensordim[6],
                                         srctermTensordim[7],
                                         srctermTensordim[8]}, options);


            std::array<MultiFab, AMREX_SPACEDIM> source_termsTrimmed;
            TrimSourceMultiFab(args ..., dmap, ba, source_termsTrimmed);
            // Convert Std::array<MultiFab,AMREX_SPACEDIM > to
            // std::array<torch::tensor, AMREX_SPACEDIM>
            Convert_StdArrMF_To_StdArrTensor(source_termsTrimmed, RHSTensor);


            // Appropriately flatten input
            torch::Tensor SrcTensorFlatx,
                          SrcTensorFlaty,
                          SrcTensorFlatz,
                          SrcTensorFlat,
                          PressureFlat,
                          bp;

            torch::Tensor presTensorTemp = torch::zeros({presTensordim[0],
                                                         presTensordim[1],
                                                         presTensordim[2]},
                                                         options);

            PressureFlat = presTensorTemp.reshape({1, 1, -1});

            SrcTensorFlatx = RHSTensor[0].reshape({1, 1, 1, -1});
            SrcTensorFlaty = RHSTensor[1].reshape({1, 1, 1, -1});
            SrcTensorFlatz = RHSTensor[2].reshape({1, 1, 1, -1});

            SrcTensorFlat = torch::cat({SrcTensorFlatx, SrcTensorFlaty}, -1);
            SrcTensorFlat = torch::cat({SrcTensorFlat,  SrcTensorFlatz}, -1);
            bp = torch::zeros({1, 1, 1, PressureFlat.size(-1)}, options);
            SrcTensorFlat = torch::cat({SrcTensorFlat, bp}, -1);


            // Get prediction as tensor
            torch::Tensor ModelOut = NETPres->forward(
                    SrcTensorFlat, srctermTensordim,presTensordim,umacTensordims
                );

            // Extract and reshape flattended pressure output:
            // 1. Pickout pressure
            torch::Tensor presTensor = ModelOut.index({
                    Slice(), Slice(), Slice(),
                    Slice(-presTensordim[0]*presTensordim[1]*presTensordim[2] - 1, -1)
                });
            // 2. unflatten
            presTensor = presTensor.reshape({1,
                                             presTensordim[0],
                                             presTensordim[1],
                                             presTensordim[2]});

            // Extract and reshape flattended staggered velocity output
            std::array<torch::Tensor, AMREX_SPACEDIM> umacTensor;
            int umacXFlatLength = umacTensordims[0]*umacTensordims[1]*umacTensordims[2];
            int umacYFlatLength = umacTensordims[3]*umacTensordims[4]*umacTensordims[5];
            int umacZFlatLength = umacTensordims[6]*umacTensordims[7]*umacTensordims[8];
            // 1. Pickout umacx
            umacTensor[0] = ModelOut.index({
                    Slice(), Slice(), Slice(),
                    Slice(0, umacXFlatLength)});
            // 2. Pickout umacy
            umacTensor[1] = ModelOut.index({
                    Slice(), Slice(), Slice(),
                    Slice(umacXFlatLength, umacXFlatLength + umacYFlatLength)});
            // 3. Pickout umacz
            umacTensor[2] = ModelOut.index({
                    Slice(), Slice(), Slice(),
                    Slice(umacXFlatLength +umacYFlatLength,
                          umacXFlatLength + umacYFlatLength+umacZFlatLength)});
            // 4. uflatten umacx
            umacTensor[0] = umacTensor[0].reshape({1,
                                                   umacTensordims[0],
                                                   umacTensordims[1],
                                                   umacTensordims[2]});
            // 5. unflatten umacy
            umacTensor[1] = umacTensor[1].reshape({1,
                                                   umacTensordims[3],
                                                   umacTensordims[4],
                                                   umacTensordims[5]});
            // 6. unflatten umacz
            umacTensor[2] = umacTensor[2].reshape({1,
                                                   umacTensordims[6],
                                                   umacTensordims[7],
                                                   umacTensordims[8]});


            /* Convert tensor to multifab using distribution map of original pressure MultiFab */
            TensorToMultifab(presTensor, pres);

            /* Convert std::array tensor to std::array multifab using distribution map of original pressure MultiFab */
            stdArrTensorTostdArrMultifab(umacTensor, umac);
        }

        /* Evaluate wrapped function with either the NN prediction or original input */
        if(RefineSol == false and step>initNum) {
            Real step_strt_time = ParallelDescriptor::second();

            // func(umac,pres,Unpack_flux(args...),Unpack_sourceTerms(args...),
            //     Unpack_alpha_fc(args...),Unpack_beta(args...),Unpack_gamma(args...),Unpack_beta_ed(args...),
            //     Unpack_geom(args...),Unpack_dt(args...));

            ap.apply(func);

            Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
            ParallelDescriptor::ReduceRealMax(step_stop_time);

            TimeDataWindow[WindowIdx]= step_stop_time; /* Add time to window of values */
            update_TimeDataWindow(args..., TimeDataWindow);



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


        }else if (RefineSol == true and TimeDataWindow[WindowIdx]>MovingAvg(TimeDataWindow))
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




    while (step < 6000)
    {
        // Spread forces to RHS
        std::array<MultiFab, AMREX_SPACEDIM> source_terms;
        for (int d=0; d<AMREX_SPACEDIM; ++d){
            source_terms[d].define(convert(ba, nodal_flag_dir[d]), dmap, 1, 6);
            source_terms[d].setVal(0.);
        }


        RealVect f_0 = RealVect{dis(gen), dis(gen), dis(gen)};
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
                        presCollect,RHSCollect,umacCollect,step,TimeDataWindow /* ML */
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
                        presCollect,RHSCollect,umacCollect,step,TimeDataWindow /* ML */
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
