#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

const int numRows=20;
const int numCols=20;
const int numSamples=304;


/* Note : Assuming executable is run from cmake "build" subdirectory. Otherwise adjust
          location where data is read from. */




/* Input: pytorch "device" object (speicifies cpu or GPU)
   Output: 3D pytorch tensor b[i][j][k] where first index provides a 2D RHS field
           i.e b[index] provides a 2D tensor read from data file */
torch::Tensor read_data_b(torch::Device device)
{
  double bdat[numSamples][numRows][numCols];

  // Read data into 3D array. 
  // The first index corresponding to file is fixed, and then we loop through 2D data file
  for (int k = 0; k < numSamples; k++) 
  {
    std::ostringstream numExample;
    numExample << std::setfill('0') << std::setw(4) << k;
    std::string numExampleString = numExample.str();

    std::string b_loc = "../../DataOuput2D/bdata/b"; 
    b_loc = b_loc+numExampleString+".dat";
    std::ifstream in1(b_loc.c_str());

    for (int i = 0; i < numRows; i++) 
    {
        for (int j = 0; j < numCols; j++) 
        {
            in1 >> bdat[k][i][j];
        }
    }
  }
 // Initizialize on GPU
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA).requires_grad(false);

  // One way of coppying over entire array of doubles
  // torch::Tensor bTensor = torch::from_blob(bdat, {numSamples,numRows, numCols}, options);  

  // Another way of copying over entire array of doubles to pytorch tensor
  torch::Tensor bTensor = torch::zeros({numSamples,numRows, numCols},options); 
  for (int i = 0; i < numSamples; i++) 
  {
    for (int j = 0; j < numRows; j++) 
    {
      for (int k = 0; k < numCols; k++) 
      {
        bTensor[i][j][k] = bdat[i][j][k];
      }
    }
  }
  return bTensor;
};


/* Input: pytorch "device" object (speicifies cpu or GPU)
   Output: 3D pytorch tensor b[i][j][k] where first index provides a 2D Solution field
           i.e b[index] provides a 2D tensor read from data file */
torch::Tensor read_data_sol(torch::Device device)
{
  // Read data into 3D array. 
  // The first index corresponding to file is fixed, and then we loop through 2D data file
  double soldat[numSamples][numRows][numCols];
  for (int k = 0; k < numSamples; k++) 
  {
    std::ostringstream numExample;
    numExample << std::setfill('0') << std::setw(4) << k;
    std::string numExampleString = numExample.str();

    std::string sol_loc = "../../DataOuput2D/soldata/res";
    sol_loc = sol_loc+numExampleString+".dat";
    std::ifstream in2(sol_loc.c_str());
    for (int i = 0; i < numRows; i++) 
    {
        for (int j = 0; j < numCols; j++) 
        {
            in2 >> soldat[k][i][j];
        }
    }
  }
  // Initizialize on GPU
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA).requires_grad(false); 

  // // One way of coppying over entire array of doubles
  // torch::Tensor SolTensor = torch::from_blob(soldat, {numSamples,numRows, numCols}, options); 

  // Another way of copying over entire array of doubles
  torch::Tensor SolTensor = torch::zeros({numSamples,numRows, numCols},options);
  for (int i = 0; i < numSamples; i++) 
        {
          for (int j = 0; j < numRows; j++) 
          {
            for (int k = 0; k < numCols; k++) 
            {
              SolTensor[i][j][k] = soldat[i][j][k];
            }
          }
        }
  return SolTensor;
};



/* It's easiest to work with pytorch dataloader objects.
   To work with these dataloader objects, we need to define a 
   pytoch dataset class that we then pass to the dataloader */
// More info:  https://pytorch.org/tutorials/advanced/cpp_frontend.html#moving-to-the-gpu
// https://discuss.pytorch.org/t/custom-dataloader/81874
// https://krshrimali.github.io/Training-Network-Using-Custom-Dataset-PyTorch-CPP/
// https://discuss.pytorch.org/t/custom-dataloader/81874/3
class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        torch::Tensor bTensor, SolTensor;

    public:
        CustomDataset(torch::Device device)
        {
          bTensor = read_data_b(device);
          SolTensor = read_data_sol(device);
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





// Define our neural network
/* The approach used here uses the "hidden" reference semantics where std::shared_ptr<MyModule> is 
   essentially hidden from the user. Essentially, a module named "MYNET" to be used is called
   "MYNETImpl" instead. Then, TORCH_MODULE defines the actual class "MYNET" to be used. The
   "generated" class is essentially a wrapper over  std::shared_ptr<LinearImpl>.  
   More info: https://pytorch.org/tutorials/advanced/cpp_frontend.html */
struct CNN_NetImpl : torch::nn::Module
{
  CNN_NetImpl(int64_t Dim ) : 
        conv1(torch::nn::Conv2dOptions(1, 1, {9,9}).stride(1).padding({4,4}).bias(false)),
        conv2(torch::nn::Conv2dOptions(1, 1, {7,7}).stride(1).padding({3,3}).bias(false)),
        lin1(torch::nn::LinearOptions(Dim*Dim,Dim*Dim).bias(false)),
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
   x = x.reshape({Current_batchsize, 1,-1}); // flatten
   x = lin1(x);
   x = x.reshape({Current_batchsize,Dim,Dim}); // unflatten
   return x;
   // std::cout << x.sizes() << std::endl;
 }

 torch::nn::Conv2d conv1,conv2;
 torch::nn::Linear lin1;
 torch::nn::LeakyReLU relu1;

};
TORCH_MODULE(CNN_Net);






int main(int argc, const char* argv[]) {




  // Define default pytorch device to be CPU and check for GPU. We use the GPU if there is one.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) 
  {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  else
  {
    std::cout << "CUDA NOT FOUND" << std::endl;
  }

  
  // Instantiate our dataset object and stack examples as a single tensor along first dimension
  auto custom_dataset = CustomDataset(device).map(torch::data::transforms::Stack<>());

  // Define model and move to GPU
  CNN_Net PoissonCNN(numRows);
  PoissonCNN->to(device);

  // Define Optimizer
  torch::optim::Adagrad optimizer(PoissonCNN->parameters(), torch::optim::AdagradOptions(0.01));
  // torch::optim::SGD optimizer(PoissonCNN->parameters(), torch::optim::SGDOptions(0.01));
  // torch::optim::Adam optimizer(PoissonCNN->parameters(), torch::optim::AdamOptions(0.001));



  int64_t batch_size = 32;
  float e1 = 1e-5;
  int64_t epoch = 0;
  int64_t numEpoch = 10000; 
  float loss = 10.0;


  /* Now, we create a data loader object and pass dataset. Note this returns a std::unique_ptr of the correct type that depends on the
  dataset, type of sampler, etc */
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),batch_size); // random batches
  // auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset),batch_size); // Sequential batches

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
            torch::Tensor output = PoissonCNN->forward(data.to(torch::kFloat32),numRows);

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

  //Write model weights
  torch::save(PoissonCNN, "CNN_Model_20x20.pt");


}
