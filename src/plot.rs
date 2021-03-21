use plotters::prelude::*;
use failure::Error;

/// Returns the maximum value of a f64 vector
pub fn vec_max(vals: &Vec<f64>) -> f64 {
    let mut m = vals[0];
    for v in vals {
        if *v > m {
            m = *v;
        }
    }
    m
}

/// Return the minimum value of an f64 vector
pub fn vec_min(vals: &Vec<f64>) -> f64 {
    let mut m = vals[0];
    for v in vals {
        if *v < m {
            m = *v;
        }
    }
    m
}

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

/// converts the feature representation into plottable values.
/// Also returns minimum and maximum values of each feature
pub(crate) fn prepare_series(series: &Vec<Vec<f64>>) -> (Vec<Vec<(f64, f64)>>, Vec<f64>, Vec<f64>) {
    let num_features: usize = series[0].len();
    let mut outs: Vec<Vec<(f64, f64)>> = vec![vec![(0.0, 0.0); series.len()]; num_features];
    let mut mins: Vec<f64> = series[0].clone();
    let mut maxs: Vec<f64> = series[0].clone();

    for s in series {
        let (s, min, max) = prepare_vec(s);
        outs.push(s);
        mins.push(min);
        maxs.push(max);
    }

    (outs, mins, maxs)
}

/// Plot multiple series on one chart
pub fn plot_multiple_series(
    series: &Vec<Vec<f64>>,
    filename: &str,
    resolution: (u32, u32),
) -> Result<(), Error> {
    let series_len: usize = series.len();
    let num_datapoints: usize = series[0].len();
    let (series, s_mins, s_maxs) = prepare_series(&series);
    let mut s_min = vec_min(&s_mins);
    let mut s_max = vec_max(&s_maxs);
    if s_min == s_max {
        // so that plotting does not get stuck
        s_min -= (s_min * 0.05).abs();
        s_max += (s_max * 0.05).abs();
    }

    let root_area = BitMapBackend::new(filename, (resolution.0, resolution.1)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let root_area = root_area
        .titled(filename, ("sans-serif", 20).into_font())
        .unwrap();

    let mut cc0 = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("value", ("sans-serif", 20).into_font())
        .build_cartesian_2d(0f64..num_datapoints as f64, s_min..s_max)?;

    cc0.configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()?;

    let colors: Vec<&'static RGBColor> =
        vec![&BLACK, &RED, &GREEN, &BLUE, &YELLOW, &CYAN, &MAGENTA];
    for (i, s) in series.iter().enumerate() {
        let c = colors[i % colors.len()];
        cc0.draw_series(LineSeries::new(s.clone(), c))?
            .label(format!("{}", i));
    }

    Ok(())
}
