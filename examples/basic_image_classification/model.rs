use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(28 * 28, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // println!(
        //     "in forward, batch_size: {} height: {} width: {} ",
        //     batch_size, height, width
        // ); // batch_size: 64 height: 28 width: 28

        // Create a channel.
        let x = images.reshape([batch_size, 1, height, width]);

        // collapse 3 dim into 1 (so we've 64 batch size and 784 entries per batch)
        let x = x.reshape([batch_size, 1 * 28 * 28]); // 64, 784 (784 is 1 * 28 * 28)
        let x = self.linear1.forward(x);
        let x = self.linear2.forward(x); // [batch_size, num_classes]
        self.activation.forward(x)
    }
}
