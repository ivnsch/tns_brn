// get minst training data
// fashion_mnist = tf.keras.datasets.fashion_mnist
// (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

mod data;
mod inference;
mod model;
mod training;

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};
use model::ModelConfig;
use training::TrainingConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/guide";

    // Train the model
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    // Infer the model
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
