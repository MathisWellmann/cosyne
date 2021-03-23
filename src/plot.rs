use failure::Error;
use plotters::prelude::*;

// prepare_vec returns a 2d vector suitable for plotting and also min, max values of input vector
pub(crate) fn prepare_vec(vals: &Vec<f64>) -> (Vec<(f64, f64)>, f64, f64) {
    let mut out = vec![(0.0, 0.0); vals.len()];
    let mut min = vals[0];
    let mut max = vals[0];

    for (i, v) in vals.iter().enumerate() {
        out[i] = (i as f64, *v);
        if *v > max {
            max = *v
        } else if *v < min {
            min = *v
        }
    }
    return (out, min, max);
}

pub fn plot_values(vals: &Vec<f64>, filename: &str, resolution: (u32, u32)) -> Result<(), Error> {
    let (vec2d, mut min, mut max) = prepare_vec(vals);
    if min == max {
        // so that plotting does not get stuck
        min -= (min * 0.05).abs();
        max += (max * 0.05).abs();
    }

    // plot the resulting function
    let root = BitMapBackend::new(filename, resolution).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(filename, ("sans-serif", 50).into_font())
        .margin(40)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..vec2d.len() as f64, min..max)?;

    chart
        .configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    chart
        .draw_series(LineSeries::new(vec2d, &BLACK))?
        .label(filename);

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
