use image::{imageops::FilterType, ImageBuffer, ImageFormat, Rgba};
use plotters::coord::Shift;
use plotters::prelude::*;
use std::fs::{remove_file, File};
use std::io::BufReader;

use crate::mnist_fashion::MnistItem;

const IMG_OUT_DIR: &str = "./img_out/";

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
const CAPTION_SIZE: i32 = 20;
const CAPTION_FONT: &str = "sans-serif";

pub fn bitmap_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    item: &PredictedItem,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let true_name = CLASS_NAMES[item.true_label() as usize];
    let predicted_name = CLASS_NAMES[item.predicted_label as usize];

    let caption = format!(
        "{} {}% ({})",
        predicted_name, item.prediction_percentage, true_name
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(
            caption,
            (
                CAPTION_FONT,
                CAPTION_SIZE,
                if item.correct_prediction() {
                    &BLUE
                } else {
                    &RED
                },
            ),
        )
        .margin(5)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    let (w, h) = chart.plotting_area().dim_in_pixel();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    let reader = to_reader(&item.item.image);

    let image = image::load(reader, ImageFormat::Png)?.resize_exact(
        w - w / 10,
        h - h / 10,
        FilterType::Nearest,
    );

    let elem: BitMapElement<_> = ((0.05, 0.95), image).into();

    chart.draw_series(std::iter::once(elem)).unwrap();
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    Ok(())
}

pub fn bitmap(item: &MnistItem) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}{}", IMG_OUT_DIR, "./item.png");
    let root = BitMapBackend::new(&path, (400, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let true_name = CLASS_NAMES[item.label as usize];

    let mut chart = ChartBuilder::on(&root)
        .caption(true_name, (CAPTION_FONT, CAPTION_SIZE))
        .margin(5)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    let (w, h) = chart.plotting_area().dim_in_pixel();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    let reader = to_reader(&item.image);

    let image = image::load(reader, ImageFormat::Png)?.resize_exact(
        w - w / 10,
        h - h / 10,
        FilterType::Nearest,
    );

    let elem: BitMapElement<_> = ((0.05, 0.95), image).into();

    chart.draw_series(std::iter::once(elem)).unwrap();
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    Ok(())
}

pub fn simple_bitmap_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    item: MnistItem,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let true_name = CLASS_NAMES[item.label as usize];

    let mut chart = ChartBuilder::on(&root)
        .caption(true_name, (CAPTION_FONT, CAPTION_SIZE))
        .margin(5)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .unwrap();

    let (w, h) = chart.plotting_area().dim_in_pixel();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    let reader = to_reader(&item.image);

    let image = image::load(reader, ImageFormat::Png)?.resize_exact(
        w - w / 10,
        h - h / 10,
        FilterType::Nearest,
    );

    let elem: BitMapElement<_> = ((0.05, 0.95), image).into();

    chart.draw_series(std::iter::once(elem)).unwrap();
    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    Ok(())
}

fn to_reader(image: &[[f32; 28]; 28]) -> BufReader<File> {
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
    let path = "./output_tmp.png";
    img_buffer.save(path).unwrap();
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    // not 100% sure it's valid to remove the file here, but working so far
    // also, remove unwrap()
    remove_file(path).unwrap();
    reader
}

pub fn bars_percentages_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    stats: &[f32],
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let data_ints: Vec<i32> = stats
        .into_iter()
        .map(|f: &f32| (f * 100.0) as i32)
        .collect();

    bars_with_root(root, &data_ints);

    Ok(())
}

pub fn bars_with_root<DB>(
    root: DrawingArea<DB, Shift>,
    data: &[i32],
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
{
    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d((0..10).into_segmented(), 0..100)
        .unwrap();

    ctx.configure_mesh()
        // attempts at showing class name instead of number
        // problems found:
        // - can rotate text only 90 degrees, not e.g 45?
        // - rotation pivot seems to be center of text, not start, so long labels will have start inside of chart
        // .x_label_formatter(&|d| {
        //     println!("\x1b[93m??? d: {:?}\x1b[0m", d);
        //     if let SegmentValue::CenterOf(v) = *d {
        //         let index = v as usize;
        //         println!("\x1b[93m??? index: {:?}\x1b[0m", index);
        //         if index < CLASS_NAMES.len() {
        //             CLASS_NAMES[index].to_string()
        //         } else {
        //             "aa".to_string()
        //         }
        //     } else {
        //         "qwqw".to_string()
        //     }
        // })
        // .x_label_style(
        //     ("sans-serif", 10)
        //         .into_font()
        //         .transform(FontTransform::Rotate90),
        // )
        .draw()
        .unwrap();

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

pub fn bitmap_with_stats(item: &PredictedItem) {
    let path = format!("{}{}", IMG_OUT_DIR, "./item_with_stats.png");
    let root = BitMapBackend::new(&path, (800, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let (left, right) = root.split_horizontally((100).percent());

    bitmap_with_root(left, item);
    bars_percentages_with_root(right, &item.stats);
}
pub fn bitmap_and_bars_with_root<DB>(root: DrawingArea<DB, Shift>, item: &PredictedItem)
where
    DB: DrawingBackend,
{
    // let root = BitMapBackend::new("./bitmap_and_bars.png", (800, 400)).into_drawing_area();
    // root.fill(&WHITE).unwrap();
    let (left, right) = root.split_horizontally((100).percent());

    bitmap_with_root(left, item);
    bars_percentages_with_root(right, &item.stats);
}

pub fn bitmap_grid(items: &[MnistItem]) {
    let path = format!("{}{}", IMG_OUT_DIR, "./items_grid.png");
    let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.split_evenly((5, 5));
    for (id, area) in areas.into_iter().enumerate() {
        // area.fill(&Palette99::pick(id)).unwrap();
        simple_bitmap_with_root(area, items.get(id).unwrap().clone());
    }
}

pub fn bitmap_and_stats_grid(items: &[PredictedItem]) {
    let path = format!("{}{}", IMG_OUT_DIR, "./items_with_stats_grid.png");
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let areas = root.margin(5, 5, 40, 40).split_evenly((5, 3));
    for (id, area) in areas.into_iter().enumerate() {
        // area.fill(&Palette99::pick(id)).unwrap();
        bitmap_and_bars_with_root(area, items.get(id).unwrap().clone());
        // simple_bitmap_with_root(area, items.get(id).unwrap().clone());
    }
}

#[derive(Debug)]
pub struct PredictedItem {
    pub item: MnistItem,
    pub stats: Vec<f32>,
    pub predicted_label: u8,
    pub prediction_percentage: u8,
}

impl PredictedItem {
    pub fn new(
        item: MnistItem,
        stats: Vec<f32>,
        predicted_label: u8,
        prediction_percentage: u8,
    ) -> PredictedItem {
        PredictedItem {
            item,
            stats,
            predicted_label,
            prediction_percentage,
        }
    }

    fn true_label(&self) -> u8 {
        self.item.label
    }

    fn correct_prediction(&self) -> bool {
        self.true_label() == self.predicted_label
    }
}
