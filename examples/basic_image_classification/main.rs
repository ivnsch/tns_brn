// get minst training data
// fashion_mnist = tf.keras.datasets.fashion_mnist
// (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

mod data;
mod inference;
mod model;
mod plot;
mod training;

use std::sync::Arc;

use burn::{
    backend::{Autodiff, Candle, NdArray, Wgpu},
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, Progress},
        dataset::{vision::MnistDataset, Dataset},
    },
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use data::MnistBatcher;
use inference::infer;
use model::ModelConfig;
use plot::{bars, plot};
use training::TrainingConfig;

struct EmptyDataLoader {}

use burn::data::dataloader::DataLoaderIterator;

impl<O: 'static> DataLoader<O> for EmptyDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn burn::data::dataloader::DataLoaderIterator<O> + 'a> {
        Box::new(EmptyIterator::new())
    }

    fn num_items(&self) -> usize {
        0
    }
}

struct EmptyIterator<O> {
    phantom: std::marker::PhantomData<O>,
}

impl<O> EmptyIterator<O> {
    fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<O> Iterator for EmptyIterator<O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl<O> DataLoaderIterator<O> for EmptyIterator<O> {
    fn progress(&self) -> burn::data::dataloader::Progress {
        Progress {
            items_processed: 0,
            items_total: 0,
        }
    }
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        // .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, Arc::new(EmptyDataLoader {}));

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = "./artifacts";

    // bars();

    type MyBackend = Candle<f32, i64>;
    // type MyBackend = NdArray<f32, i8>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    // let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = burn::backend::ndarray::NdArrayDevice::default();
    let device = burn::backend::candle::CandleDevice::Metal(0);
    let config = TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new());

    let data_set = MnistDataset::train();
    let first = data_set.get(0);
    println!("\x1b[93m??? first: {:?}\x1b[0m", first);

    plot(first.unwrap())?;

    train::<MyAutodiffBackend>(&artifact_dir, config, device.clone());

    infer::<MyAutodiffBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );

    Ok(())
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
