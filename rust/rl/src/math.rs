/// For compatibility with python max.
pub fn max(xs: &[Option<f32>]) -> (usize, Option<f32>) {
    let ret = xs.iter().rev().enumerate().max_by(|&a, &b| match (a, b) {
        ((_, Some(a)), (_, Some(b))) => a.total_cmp(b),
        ((_, Some(_)), (_, None)) => std::cmp::Ordering::Greater,
        ((_, None), (_, Some(_))) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    });

    if let Some(ret) = ret {
        (xs.len() - 1 - ret.0, *ret.1)
    } else {
        unimplemented!("0 length collections are not supported.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn non_empty_all_none() {
        let m = max(&[None, None, None]);
        assert_eq!(m.0, 0);
        assert_eq!(m.1, None);
    }

    #[test]
    fn non_empty_some_none() {
        let m = max(&[None, Some(1.0), None]);
        assert_eq!(m.0, 1);
        assert_float_eq!(m.1.unwrap(), 1.0, rmax <= 1e10);
    }

    #[test]
    fn non_empty() {
        let m = max(&[Some(1.0), Some(7.0), Some(0.9), Some(1.0), Some(7.0)]);
        assert_eq!(m.0, 1);
        assert_float_eq!(m.1.unwrap(), 7.0, rmax <= 1e10);
    }
}
