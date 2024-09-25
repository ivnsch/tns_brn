// get minst training data
// fashion_mnist = tf.keras.datasets.fashion_mnist
// (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

mod data;
mod mnist_fashion;
mod model;
mod plot;
mod training;

use std::sync::Arc;

use burn::{
    backend::{
        // candle::{CandleDevice, MetalDevice},
        candle::CandleDevice,
        wgpu::WgpuDevice,
        Autodiff,
        Candle,
        NdArray,
        Wgpu,
    },
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoader, DataLoaderBuilder, Progress},
        dataset::Dataset,
    },
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::AutodiffBackend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use data::MnistBatcher;
use mnist_fashion::{MnistDataset, MnistItem};
use model::{Model, ModelConfig, ModelRecord};
use plot::{bitmap, bitmap_and_bars, bitmap_and_stats_grid, bitmap_grid, PredictedItem};
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: &TrainingConfig, device: &B::Device) {
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
        .metric_train_numeric(LossMetric::new())
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
// type MyBackend = Wgpu<f32, i32>;
type MyBackend = Candle<f32, u32>;
// type MyBackend = NdArray<f32, i8>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact_dir = "./artifacts";

    // bars();

    // let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = burn::backend::ndarray::NdArrayDevice::default();
    let device = CandleDevice::metal(0);
    let config = TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new());

    let data_set = MnistDataset::train();
    let first = data_set.get(0);
    println!("\x1b[93m??? first: {:?}\x1b[0m", first);

    bitmap(first.unwrap())?;

    train::<MyAutodiffBackend>(&artifact_dir, &config, &device);

    let item = MnistDataset::test().get(12).unwrap();

    // infer::<MyAutodiffBackend>(artifact_dir, device, item);

    // evaluate on 1 item

    // let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
    //     .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let predicted_item = test(&item, &config, &device, record);

    bitmap_and_bars(&predicted_item);

    let items = MnistDataset::test().iter().take(25).collect();
    bitmap_grid(items);

    let predicted_items = MnistDataset::test()
        .iter()
        .take(15)
        .map(|i| test(&i, artifact_dir, &config, &device))
        .collect();
    bitmap_and_stats_grid(predicted_items);

    Ok(())
}

fn test(
    item: &MnistItem,
    config: &TrainingConfig,
    device: &CandleDevice,
    model_record: ModelRecord<MyAutodiffBackend>,
) -> PredictedItem {
    let model: Model<MyAutodiffBackend> = config.model.init(device).load_record(model_record);

    let batcher = MnistBatcher::new(device.clone());
    let batch = batcher.batch(vec![item.clone()]);
    let output = model.forward(batch.images);
    let output = softmax(output, 1);
    println!("output: {}", output);

    let output_floats = output.to_data().convert::<f32>().to_vec().unwrap();
    println!("output_floats: {:?}", output_floats);
    // bars_percentages(output_floats).unwrap();

    let predicted = output.clone().argmax(1).flatten::<1>(0, 1).into_scalar();
    let predicted_percentage_floats = output.max().into_data().convert::<f32>().to_vec().unwrap();
    let predicted_percentage: f32 = predicted_percentage_floats[0];
    println!("predicted_percentage: {}", predicted_percentage);
    let predicted_percentage_int = (predicted_percentage * 100.0) as u8;

    PredictedItem {
        item: item.clone(),
        stats: output_floats,
        predicted_label: predicted as u8,
        prediction_percentage: predicted_percentage_int,
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
