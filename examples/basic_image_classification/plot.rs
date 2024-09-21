use image::{imageops::FilterType, ImageBuffer, ImageFormat, Rgba};
use plotters::coord::Shift;
use plotters::prelude::*;
use std::fs::{remove_file, File};
use std::io::BufReader;

use crate::mnist_fashion::MnistItem;

const OUT_FILE_NAME: &str = "./mybitmap.png";
const CLASS_NAMES: [&str; 10] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

pub fn plot(item: MnistItem) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    bitmap_with_root(root, item);

    Ok(())
}

pub fn bitmap_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    item: MnistItem,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let mut chart = ChartBuilder::on(&root)
        .caption(CLASS_NAMES[item.label as usize], ("sans-serif", 30))
        .margin(5)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    let (w, h) = chart.plotting_area().dim_in_pixel();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    let reader = to_reader(item);

    let image = image::load(reader, ImageFormat::Png)?.resize_exact(
        w - w / 10,
        h - h / 10,
        FilterType::Nearest,
    );

    let elem: BitMapElement<_> = ((0.05, 0.95), image).into();

    chart.draw_series(std::iter::once(elem)).unwrap();
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

fn to_reader(item: MnistItem) -> BufReader<File> {
    let image = item.image;

    // Convert the raw f32 data to u8 and create an image buffer
    let mut img_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(28, 28);

    for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
        let value = image[y as usize][x as usize];
        // let normalized_value = (value * 255.0).min(0.0).max(255.0) as u8;
        // println!("value: {}, normalized: {}", value, normalized_value);
        let normalized_value = value as u8;
        *pixel = Rgba([normalized_value, normalized_value, normalized_value, 255]);
        // Grayscale to RGBA
    }

    // there should be a better way to make the data a png than writing and reading from file..
    let path = "./output.png";
    img_buffer.save(path).unwrap();
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    // not 100% sure it's valid to remove the file here, but working so far
    // also, remove unwrap()
    // remove_file(path).unwrap();
    reader
}

pub fn bars_percentages(data: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("./bars.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    bars_percentages_with_root(root, data)
}

pub fn bars_percentages_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    data: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let data_ints = data.into_iter().map(|f: f32| (f * 100.0) as i32).collect();

    bars_with_root(root, data_ints);

    Ok(())
}

pub fn bars_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    data: Vec<i32>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Bar Demo", ("sans-serif", 40))
        .build_cartesian_2d((0..10).into_segmented(), 0..100)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series((0..).zip(data.iter()).map(|(x, y)| {
        let mut bar = Rectangle::new(
            [
                (SegmentValue::Exact(x), 0),
                (SegmentValue::Exact(x + 1), *y),
            ],
            GREEN.filled(),
        );
        bar.set_margin(0, 0, 5, 5);
        bar
    }))
    .unwrap();
    Ok(())
}

pub fn bitmap_and_bars(item: MnistItem, data: Vec<f32>) {
    let root = BitMapBackend::new("./bitmap_and_bars.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let (left, right) = root.split_horizontally((100).percent());

    bitmap_with_root(left, item);
    bars_percentages_with_root(right, data);
}
