// get minst training data
// fashion_mnist = tf.keras.datasets.fashion_mnist
// (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

mod data;
mod inference;
mod model;
mod plot;
mod training;

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::{vision::MnistDataset, Dataset},
    optim::AdamConfig,
};
use data::MnistBatcher;
use model::ModelConfig;
use plot::plot;
use training::TrainingConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new());

    let batcher_train = MnistBatcher::<MyAutodiffBackend>::new(device.clone());

    let data_set = MnistDataset::train();
    let first = data_set.get(0);
    println!("\x1b[93m??? first: {:?}\x1b[0m", first);

    plot(first.unwrap())?;

    // let dataloader_train = DataLoaderBuilder::new(batcher_train)
    //     .batch_size(config.batch_size)
    //     .shuffle(config.seed)
    //     .num_workers(config.num_workers)
    //     .build(MnistDataset::train());

    Ok(())
}
